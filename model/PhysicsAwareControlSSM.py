import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    """Physics-aware Control-SSM built on iTransformer encoder features."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        self.lambda_phys = getattr(configs, 'lambda_phys', 0.05)
        self.lambda_lyap = getattr(configs, 'lambda_lyap', 0.1)
        self.lambda_rho = getattr(configs, 'lambda_rho', 0.1)
        self.lyap_epsilon = getattr(configs, 'lyap_epsilon', 1e-3)
        self.energy_alpha = getattr(configs, 'energy_alpha', 0.1)

        self.rain_idx = getattr(configs, 'rain_idx', 0)
        self.flow_idx = getattr(configs, 'flow_idx', 1)
        self.wind_idx = getattr(configs, 'wind_idx', 2)

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
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
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        self.state_proj = nn.Linear(configs.d_model, configs.d_model)
        self.log_a_diag = nn.Parameter(torch.randn(configs.d_model) * 0.02)
        self.control_proj = nn.Linear(4, configs.d_model)
        self.control_gate = nn.Linear(4, configs.d_model)
        self.state_to_y = nn.Linear(configs.d_model, self.c_out)

        # P = L L^T + eps * I for PSD Lyapunov matrix
        self.lyap_cholesky = nn.Parameter(torch.eye(configs.d_model))

        self._cache = {}

    def _build_control_signal(self, x_enc):
        target_x = x_enc[:, :, :self.c_out]
        delta_x = target_x[:, 1:, :] - target_x[:, :-1, :]
        delta_x = delta_x.abs().mean(dim=-1, keepdim=True)

        rain = x_enc[:, 1:, self.rain_idx:self.rain_idx + 1]
        flow = x_enc[:, 1:, self.flow_idx:self.flow_idx + 1]
        wind = x_enc[:, 1:, self.wind_idx:self.wind_idx + 1]
        return torch.cat([delta_x, rain, flow, wind], dim=-1)

    def _rollout_states(self, s0, u_context):
        a_diag = torch.exp(self.log_a_diag)
        states = [s0]
        control_last = u_context[:, -1, :]

        for _ in range(self.pred_len):
            gate = torch.sigmoid(self.control_gate(control_last))
            control_term = self.control_proj(control_last) * gate
            s_next = states[-1] * a_diag + control_term
            states.append(s_next)

        return torch.stack(states, dim=1), a_diag, control_last

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        token_count = enc_out.shape[1]
        used_tokens = min(self.c_out, token_count)
        s0 = self.state_proj(enc_out[:, :used_tokens, :].mean(dim=1))

        u_context = self._build_control_signal(x_enc)
        states, a_diag, control_last = self._rollout_states(s0, u_context)
        pred = self.state_to_y(states[:, 1:, :])

        if self.use_norm:
            rescale = stdev[:, 0, :self.c_out].unsqueeze(1).repeat(1, self.pred_len, 1)
            rebias = means[:, 0, :self.c_out].unsqueeze(1).repeat(1, self.pred_len, 1)
            pred = pred * rescale + rebias

        self._cache = {
            'states': states,
            'a_diag': a_diag,
            'u_context': u_context,
            'control_last': control_last,
        }
        return pred, attns

    def compute_aux_loss(self):
        if not self._cache:
            device = self.log_a_diag.device
            return torch.tensor(0.0, device=device), {}

        states = self._cache['states']
        a_diag = self._cache['a_diag']
        u_context = self._cache['u_context']

        L = self.lyap_cholesky
        p_mat = L @ L.t() + 1e-5 * torch.eye(L.size(0), device=L.device)

        s_t = states[:, :-1, :]
        s_tp1 = states[:, 1:, :]
        v_t = torch.einsum('bth,hk,btk->bt', s_t, p_mat, s_t)
        v_tp1 = torch.einsum('bth,hk,btk->bt', s_tp1, p_mat, s_tp1)
        delta_v = v_tp1 - v_t + self.lyap_epsilon * v_t
        lyap_loss = F.relu(delta_v).mean()

        rho_loss = torch.clamp(a_diag.max() - 1.0, min=0.0)

        flow_delta = u_context[:, -1, 2].unsqueeze(1)
        rain_term = u_context[:, -1, 1].unsqueeze(1)
        mass_balance = (s_tp1 - s_t).abs().mean(dim=-1)
        mass_loss = (mass_balance - flow_delta).pow(2).mean()

        energy_tp1 = s_tp1.pow(2).mean(dim=-1)
        energy_t = s_t.pow(2).mean(dim=-1)
        energy_loss = (energy_tp1 - energy_t - self.energy_alpha * rain_term).pow(2).mean()

        physics_loss = mass_loss + energy_loss
        total_aux = self.lambda_phys * physics_loss + self.lambda_lyap * lyap_loss + self.lambda_rho * rho_loss

        stats = {
            'loss_phys': physics_loss.detach(),
            'loss_lyap': lyap_loss.detach(),
            'loss_rho': rho_loss.detach(),
        }
        return total_aux, stats

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]
