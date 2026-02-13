import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer


class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )
        self.spatial = nn.Conv1d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # x: [B, C, L]
        avg = x.mean(dim=-1)
        mx = x.max(dim=-1).values
        channel_attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).unsqueeze(-1)
        x = x * channel_attn

        avg_spatial = x.mean(dim=1, keepdim=True)
        max_spatial = x.max(dim=1, keepdim=True).values
        spatial_attn = torch.sigmoid(self.spatial(torch.cat([avg_spatial, max_spatial], dim=1)))
        return x * spatial_attn


class DiffusionGenerator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, z, t):
        t = t.unsqueeze(-1)
        return self.net(torch.cat([z, t], dim=-1))


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.periodic_k = getattr(configs, 'ddhace_periodic_k', 4)

        # iTransformer backbone
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # short-term conv + cbam
        self.conv1 = nn.Conv1d(configs.enc_in, configs.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(configs.d_model, configs.d_model, kernel_size=3, padding=1)
        self.cbam1 = CBAM1D(configs.d_model)
        self.cbam2 = CBAM1D(configs.d_model)

        # periodic embedding params
        self.amp = nn.Parameter(torch.ones(self.periodic_k))
        self.omega = nn.Parameter(torch.linspace(2 * math.pi / max(self.seq_len, 2), 2 * math.pi, self.periodic_k))
        self.phase = nn.Parameter(torch.zeros(self.periodic_k))
        self.periodic_proj = nn.Linear(self.periodic_k, configs.d_model)
        self.mu_proj = nn.Linear(configs.enc_in, configs.d_model)

        self.gate = nn.Linear(configs.d_model * 3, configs.d_model)
        self.forecast_head = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in),
        )

        self.repr_proj = nn.Linear(configs.d_model, configs.d_model)
        self.diffusion_generator = DiffusionGenerator(configs.d_model)

    def _periodic_embedding(self, B, L, device):
        t = torch.arange(L, device=device).float().unsqueeze(-1)
        periodic = self.amp * torch.sin(t * self.omega + self.phase)
        periodic = periodic.unsqueeze(0).repeat(B, 1, 1)
        return self.periodic_proj(periodic)

    def encode_features(self, x_enc, x_mark_enc):
        B, L, _ = x_enc.shape
        device = x_enc.device

        # iTransformer global context
        tokens = self.enc_embedding(x_enc, x_mark_enc)
        tokens, attns = self.encoder(tokens, attn_mask=None)
        backbone_context = tokens.mean(dim=1).unsqueeze(1).repeat(1, L, 1)

        # local conv + cbam
        z_c = x_enc.permute(0, 2, 1)
        z_c = self.cbam1(F.gelu(self.conv1(z_c)))
        z_c = self.cbam2(F.gelu(self.conv2(z_c)))
        z_c = z_c.permute(0, 2, 1)
        z_c = z_c + backbone_context

        e_t = self._periodic_embedding(B, L, device)
        mu = F.avg_pool1d(x_enc.permute(0, 2, 1), kernel_size=3, stride=1, padding=1).permute(0, 2, 1)
        mu = self.mu_proj(mu)

        z = torch.cat([z_c, e_t, mu], dim=-1)
        g = torch.sigmoid(self.gate(z))
        fused = g * e_t + (1 - g) * z_c

        return fused, attns

    def forecast(self, x_enc, x_mark_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        else:
            means, stdev = None, None

        fused, attns = self.encode_features(x_enc, x_mark_enc)
        forecast_seq = self.forecast_head(fused)
        dec_out = forecast_seq[:, -self.pred_len :, :]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        repr_vec = self.repr_proj(fused.mean(dim=1))
        return dec_out, repr_vec, attns

    def sample_diffusion_negatives(self, repr_vec, steps=4):
        B, D = repr_vec.shape
        device = repr_vec.device
        negatives = []
        base = repr_vec
        for s in range(steps):
            t = torch.full((B,), float(s + 1) / steps, device=device)
            noise = self.diffusion_generator(base, t)
            alpha = 1.0 - float(s + 1) / (steps + 1)
            neg = math.sqrt(alpha) * base + math.sqrt(max(1e-6, 1 - alpha)) * noise
            negatives.append(F.normalize(neg, dim=-1))
        return torch.stack(negatives, dim=1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_aux=False):
        dec_out, repr_vec, attns = self.forecast(x_enc, x_mark_enc)
        if self.output_attention:
            dec_out = dec_out[:, -self.pred_len :, :]
        if return_aux:
            return dec_out, {'repr': repr_vec, 'attns': attns}
        return dec_out[:, -self.pred_len :, :]
