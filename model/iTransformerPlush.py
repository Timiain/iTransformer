import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class CrossAttention(nn.Module):
    """Cross attention for branch interaction."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query_tokens, context_tokens):
        attn_out, _ = self.attn(query_tokens, context_tokens, context_tokens)
        return self.norm(query_tokens + attn_out)


class MultiScaleFFT(nn.Module):
    """Frequency-domain multi-band convolution."""

    def __init__(self, d_model, kernel_sizes=(3, 5, 7), dropout=0.1):
        super().__init__()
        in_channels = 2 * d_model
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, d_model, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, N, D]
        fft_x = torch.fft.rfft(x, dim=1)  # [B, F, D], complex
        fft_feat = torch.cat([fft_x.real, fft_x.imag], dim=-1)  # [B, F, 2D]

        feat = fft_feat.permute(0, 2, 1)  # [B, 2D, F]
        multiscale = [conv(feat) for conv in self.convs]
        freq_out = torch.stack(multiscale, dim=0).mean(dim=0)  # [B, D, F]
        freq_out = self.dropout(freq_out).permute(0, 2, 1)  # [B, F, D]

        # Align frequency tokens back to variate-token length.
        freq_out = F.interpolate(
            freq_out.permute(0, 2, 1),
            size=x.size(1),
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)

        return self.proj(freq_out)


class MultiScalePatch(nn.Module):
    """Patch branch with parallel multi-scale 1D convs."""

    def __init__(self, d_model, patch_sizes=(32, 64, 128), dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=p, padding=p // 2)
            for p in patch_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, N, D]
        token_feat = x.permute(0, 2, 1)  # [B, D, N]
        conv_outs = [conv(token_feat) for conv in self.convs]

        # In rare even-kernel boundary cases, trim to the shortest length.
        min_len = min(out.size(-1) for out in conv_outs)
        conv_outs = [out[..., :min_len] for out in conv_outs]

        patch_out = torch.stack(conv_outs, dim=0).mean(dim=0).permute(0, 2, 1)
        patch_out = self.dropout(patch_out)

        # Restore original token length if needed.
        if patch_out.size(1) != x.size(1):
            patch_out = F.interpolate(
                patch_out.permute(0, 2, 1),
                size=x.size(1),
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)

        return self.norm(patch_out)


class FFTBranch(nn.Module):
    def __init__(self, d_model, n_heads, dropout, fft_kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.ms_fft = MultiScaleFFT(d_model, kernel_sizes=fft_kernel_sizes, dropout=dropout)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)

    def forward(self, base_tokens):
        fft_tokens = self.ms_fft(base_tokens)
        return self.cross_attn(base_tokens, fft_tokens)


class PatchBranch(nn.Module):
    def __init__(self, d_model, n_heads, dropout, patch_sizes=(32, 64, 128)):
        super().__init__()
        self.ms_patch = MultiScalePatch(d_model, patch_sizes=patch_sizes, dropout=dropout)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)

    def forward(self, base_tokens):
        patch_tokens = self.ms_patch(base_tokens)
        return self.cross_attn(base_tokens, patch_tokens)


class GateFusion(nn.Module):
    """Adaptive gate fusion for FFT/Patch dual branches."""

    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, fft_tokens, patch_tokens):
        fft_pool = fft_tokens.mean(dim=1)
        patch_pool = patch_tokens.mean(dim=1)
        alpha = self.gate(torch.cat([fft_pool, patch_pool], dim=-1)).unsqueeze(1)
        fused = alpha * fft_tokens + (1.0 - alpha) * patch_tokens
        return self.norm(fused)


class Model(nn.Module):
    """iTransformer+ with FFT-Patch parallel subnetwork."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Original embedding module (kept unchanged)
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.class_strategy = configs.class_strategy

        patch_sizes = tuple(getattr(configs, "patch_sizes", [32, 64, 128]))
        fft_kernel_sizes = tuple(getattr(configs, "fft_kernel_sizes", [3, 5, 7]))

        # New parallel dual-branch subnetwork
        self.fft_branch = FFTBranch(configs.d_model, configs.n_heads, configs.dropout, fft_kernel_sizes)
        self.patch_branch = PatchBranch(configs.d_model, configs.n_heads, configs.dropout, patch_sizes)
        self.gate_fusion = GateFusion(configs.d_model)

        # Original encoder stack (kept unchanged)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
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
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, n_vars = x_enc.shape

        # 1) Variate token embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # 2) FFT/Patch parallel branches + gate fusion
        fft_out = self.fft_branch(enc_out)
        patch_out = self.patch_branch(enc_out)
        fused_out = self.gate_fusion(fft_out, patch_out)

        # Residual injection before the original encoder
        enc_out = enc_out + fused_out

        # 3) Original iTransformer encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :n_vars]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]
