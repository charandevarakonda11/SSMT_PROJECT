# import math
# import torch
# from torch import nn
# from torch.nn import functional as F

# import modules
# import common_modified
# import attentions
# import monotonic_align

# # --- Predictors ---

# class DurationPredictor(nn.Module):
#     def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
#         super().__init__()
#         self.drop = nn.Dropout(p_dropout)
#         self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
#         self.norm_1 = attentions.LayerNorm(filter_channels)
#         self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
#         self.norm_2 = attentions.LayerNorm(filter_channels)
#         self.proj = nn.Conv1d(filter_channels, 1, 1)

#     def forward(self, x, x_mask):
#         x = self.conv_1(x * x_mask)
#         x = torch.relu(x)
#         x = self.norm_1(x)
#         x = self.drop(x)
#         x = self.conv_2(x * x_mask)
#         x = torch.relu(x)
#         x = self.norm_2(x)
#         x = self.drop(x)
#         x = self.proj(x * x_mask)
#         return x * x_mask


# class PitchPredictor(nn.Module):
#     def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
#         super().__init__()
#         self.drop = nn.Dropout(p_dropout)
#         self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
#         self.norm_1 = attentions.LayerNorm(filter_channels)
#         self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
#         self.norm_2 = attentions.LayerNorm(filter_channels)
#         self.proj = nn.Conv1d(filter_channels, 1, 1)

#     def forward(self, x, x_mask):
#         x = self.conv_1(x * x_mask)
#         x = torch.relu(x)
#         x = self.norm_1(x)
#         x = self.drop(x)
#         x = self.conv_2(x * x_mask)
#         x = torch.relu(x)
#         x = self.norm_2(x)
#         x = self.drop(x)
#         x = self.proj(x * x_mask)
#         return x * x_mask


# class EnergyPredictor(nn.Module):
#     def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
#         super().__init__()
#         self.drop = nn.Dropout(p_dropout)
#         self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
#         self.norm_1 = attentions.LayerNorm(filter_channels)
#         self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
#         self.norm_2 = attentions.LayerNorm(filter_channels)
#         self.proj = nn.Conv1d(filter_channels, 1, 1)

#     def forward(self, x, x_mask):
#         x = self.conv_1(x * x_mask)
#         x = torch.relu(x)
#         x = self.norm_1(x)
#         x = self.drop(x)
#         x = self.conv_2(x * x_mask)
#         x = torch.relu(x)
#         x = self.norm_2(x)
#         x = self.drop(x)
#         x = self.proj(x * x_mask)
#         return x * x_mask

# # --- Text Encoder ---

# class TextEncoder(nn.Module):
#     def __init__(
#         self,
#         n_vocab,
#         out_channels,
#         hidden_channels,
#         filter_channels,
#         filter_channels_dp,
#         n_heads,
#         n_layers,
#         kernel_size,
#         p_dropout,
#         window_size=None,
#         block_length=None,
#         mean_only=False,
#         prenet=False,
#         gin_channels=0,
#     ):
#         super().__init__()
#         self.emb = nn.Embedding(n_vocab, hidden_channels)
#         nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

#         if prenet:
#             self.pre = modules.ConvReluNorm(
#                 hidden_channels, hidden_channels, hidden_channels,
#                 kernel_size=5, n_layers=3, p_dropout=0.5,
#             )
#         else:
#             self.pre = None

#         self.encoder = attentions.Encoder(
#             hidden_channels,
#             filter_channels,
#             n_heads,
#             n_layers,
#             kernel_size,
#             p_dropout,
#             window_size=window_size,
#             block_length=block_length,
#         )

#         self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
#         if not mean_only:
#             self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
#         else:
#             self.proj_s = None

#         # New predictors
#         self.proj_w = DurationPredictor(
#             hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout
#         )
#         self.pitch_predictor = PitchPredictor(
#             hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout
#         )
#         self.energy_predictor = EnergyPredictor(
#             hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout
#         )
#     def forward(self, x, x_lengths, g=None):
#             x = self.emb(x) * math.sqrt(self.emb.embedding_dim)  # [b, t, h]
#             x = torch.transpose(x, 1, -1)  # [b, h, t]
#             x_mask = torch.unsqueeze(common_modified.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

#             if self.pre is not None:
#                 x = self.pre(x, x_mask)
#             x = self.encoder(x, x_mask)

#             if g is not None:
#                 g_exp = g.expand(-1, -1, x.size(-1))
#                 x_dp = torch.cat([torch.detach(x), g_exp], 1)
#             else:
#                 x_dp = torch.detach(x)

#             x_m = self.proj_m(x) * x_mask

#             if self.proj_s is not None:
#                 x_logs = self.proj_s(x) * x_mask
#             else:
#                 x_logs = torch.zeros_like(x_m)

#             logw = self.proj_w(x_dp, x_mask)
#             pitch = self.pitch_predictor(x_dp, x_mask)
#             energy = self.energy_predictor(x_dp, x_mask)

#             return x_m, x_logs, logw, pitch, energy, x_mask


# class FlowSpecDecoder(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         hidden_channels,
#         kernel_size,
#         dilation_rate,
#         n_blocks,
#         n_layers,
#         p_dropout=0.0,
#         n_split=4,
#         n_sqz=2,
#         sigmoid_scale=False,
#         gin_channels=0,
#     ):
#         super().__init__()

#         self.in_channels = in_channels
#         self.hidden_channels = hidden_channels
#         self.kernel_size = kernel_size
#         self.dilation_rate = dilation_rate
#         self.n_blocks = n_blocks
#         self.n_layers = n_layers
#         self.p_dropout = p_dropout
#         self.n_split = n_split
#         self.n_sqz = n_sqz
#         self.sigmoid_scale = sigmoid_scale
#         self.gin_channels = gin_channels

#         self.flows = nn.ModuleList()
#         for b in range(n_blocks):
#             self.flows.append(modules.ActNorm(channels=in_channels * n_sqz))
#             self.flows.append(
#                 modules.InvConvNear(channels=in_channels * n_sqz, n_split=n_split)
#             )
#             self.flows.append(
#                 attentions.CouplingBlock(
#                     in_channels * n_sqz,
#                     hidden_channels,
#                     kernel_size=kernel_size,
#                     dilation_rate=dilation_rate,
#                     n_layers=n_layers,
#                     gin_channels=gin_channels,
#                     p_dropout=p_dropout,
#                     sigmoid_scale=sigmoid_scale,
#                 )
#             )

#     def forward(self, x, x_mask, g=None, reverse=False):
#         if not reverse:
#             flows = self.flows
#             logdet_tot = 0
#         else:
#             flows = reversed(self.flows)
#             logdet_tot = None

#         if self.n_sqz > 1:
#             x, x_mask = common_modified.squeeze(x, x_mask, self.n_sqz)
#         for f in flows:
#             if not reverse:
#                 x, logdet = f(x, x_mask, g=g, reverse=reverse)
#                 logdet_tot += logdet
#             else:
#                 x, logdet = f(x, x_mask, g=g, reverse=reverse)
#         if self.n_sqz > 1:
#             x, x_mask = common_modified.unsqueeze(x, x_mask, self.n_sqz)
#         return x, logdet_tot

#     def store_inverse(self):
#         for f in self.flows:
#             f.store_inverse()


# class FlowGenerator(nn.Module):
#     def __init__(
#         self,
#         n_vocab,
#         hidden_channels,
#         filter_channels,
#         filter_channels_dp,
#         out_channels,
#         kernel_size=3,
#         n_heads=2,
#         n_layers_enc=6,
#         p_dropout=0.0,
#         n_blocks_dec=12,
#         kernel_size_dec=5,
#         dilation_rate=5,
#         n_block_layers=4,
#         p_dropout_dec=0.0,
#         n_speakers=0,
#         gin_channels=0,
#         n_split=4,
#         n_sqz=1,
#         sigmoid_scale=False,
#         window_size=None,
#         block_length=None,
#         mean_only=False,
#         hidden_channels_enc=None,
#         hidden_channels_dec=None,
#         prenet=False,
#         **kwargs
#     ):

#         super().__init__()
#         self.n_vocab = n_vocab
#         self.hidden_channels = hidden_channels
#         self.filter_channels = filter_channels
#         self.filter_channels_dp = filter_channels_dp
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.n_heads = n_heads
#         self.n_layers_enc = n_layers_enc
#         self.p_dropout = p_dropout
#         self.n_blocks_dec = n_blocks_dec
#         self.kernel_size_dec = kernel_size_dec
#         self.dilation_rate = dilation_rate
#         self.n_block_layers = n_block_layers
#         self.p_dropout_dec = p_dropout_dec
#         self.n_speakers = n_speakers
#         self.gin_channels = gin_channels
#         self.n_split = n_split
#         self.n_sqz = n_sqz
#         self.sigmoid_scale = sigmoid_scale
#         self.window_size = window_size
#         self.block_length = block_length
#         self.mean_only = mean_only
#         self.hidden_channels_enc = hidden_channels_enc
#         self.hidden_channels_dec = hidden_channels_dec
#         self.prenet = prenet

#         self.encoder = TextEncoder(
#             n_vocab,
#             out_channels,
#             hidden_channels_enc or hidden_channels,
#             filter_channels,
#             filter_channels_dp,
#             n_heads,
#             n_layers_enc,
#             kernel_size,
#             p_dropout,
#             window_size=window_size,
#             block_length=block_length,
#             mean_only=mean_only,
#             prenet=prenet,
#             gin_channels=gin_channels,
#         )

#         self.decoder = FlowSpecDecoder(
#             out_channels,
#             hidden_channels_dec or hidden_channels,
#             kernel_size_dec,
#             dilation_rate,
#             n_blocks_dec,
#             n_block_layers,
#             p_dropout=p_dropout_dec,
#             n_split=n_split,
#             n_sqz=n_sqz,
#             sigmoid_scale=sigmoid_scale,
#             gin_channels=gin_channels,
#         )

#         if n_speakers > 1:
#             self.emb_g = nn.Embedding(n_speakers, gin_channels)
#             nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

   
#     def forward(
#         self,
#         x,
#         x_lengths,
#         y=None,
#         y_lengths=None,
#         pitch=None,
#         energy=None,
#         g=None,
#         gen=False,
#         noise_scale=1.0,
#         length_scale=1.0,
#     ):
#         if g is not None:
#             g = F.normalize(self.emb_g(g)).unsqueeze(-1)  # [b, h]
#         # x_m, x_logs, logw, x_mask = self.encoder(x, x_lengths, g=g)
#         x_m, x_logs, logw, pitch, energy, x_mask = self.encoder(x, x_lengths, g=g)

#         print(f"Shape of pitch from encoder: {pitch.shape}")  # [b, 1, t_enc] where t_enc is usually x.size(1)
#         print(f"Shape of energy from encoder: {energy.shape}") # [b, 1, t_enc]
#         print(f"Shape of x_mask: {x_mask.shape}")      # [b, 1, t_enc]

#         if gen:
#             w = torch.exp(logw) * x_mask * length_scale
#             w_ceil = torch.ceil(w)
#             y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
#             y_max_length = None
#         else:
#             y_max_length = y.size(2)
#         y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
#         z_mask = torch.unsqueeze(common_modified.sequence_mask(y_lengths, y_max_length), 1).to(
#             x_mask.dtype
#         )
#         attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

#         print(f"Shape of y: {y.shape}")             # [b, n_mel, t_dec]
#         print(f"Shape of z_mask: {z_mask.shape}")   # [b, 1, t_dec]
#         print(f"Shape of attn_mask: {attn_mask.shape}") # [b, 1, t_dec, t_enc]

#         if gen:
#             attn = common_modified.generate_path(
#                 w_ceil.squeeze(1), attn_mask.squeeze(1)
#             ).unsqueeze(1)
#             z_m = torch.matmul(
#                 attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)
#             ).transpose(
#                 1, 2
#             )  # [b, t_dec, t_enc], [b, t_enc, d] -> [b, d, t_dec]
#             z_logs = torch.matmul(
#                 attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)
#             ).transpose(
#                 1, 2
#             )  # [b, t_dec, t_enc], [b, t_enc, d] -> [b, d, t_dec]
#             logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

#             # Project pitch and energy using attention during generation (optional, might need different logic)
#             pred_pitch = torch.matmul(attn.squeeze(1), pitch.squeeze(1).unsqueeze(-1)).unsqueeze(1)
#             pred_energy = torch.matmul(attn.squeeze(1), energy.squeeze(1).unsqueeze(-1)).unsqueeze(1)

#             z = (z_m + torch.exp(z_logs) * torch.randn_like(z_m) * noise_scale) * z_mask
#             y, logdet = self.decoder(z, z_mask, g=g, reverse=True)
#             return (
#                 (y, z_m, z_logs, logdet, z_mask),
#                 (x_m, x_logs, x_mask),
#                 (attn, logw, logw_),
#                 pred_pitch,
#                 pred_energy,
#             )
#         else:
#             z, logdet = self.decoder(y, z_mask, g=g, reverse=False)
#             with torch.no_grad():
#                 x_s_sq_r = torch.exp(-2 * x_logs)
#                 logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(
#                     -1
#                 )  # [b, t_enc, 1]
#                 logp2 = torch.matmul(
#                     x_s_sq_r.transpose(1, 2), -0.5 * (z ** 2)
#                 )  # [b, t_enc, d] x [b, d, t_dec] = [b, t_enc, t_dec]
#                 logp3 = torch.matmul(
#                     (x_m * x_s_sq_r).transpose(1, 2), z
#                 )  # [b, t_enc, d] x [b, d, t_dec] = [b, t_enc, t_dec]
#                 logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(
#                     -1
#                 )  # [b, t_enc, 1]
#                 logp = logp1 + logp2 + logp3 + logp4  # [b, t_enc, t_dec]

#                 attn = (
#                     monotonic_align.maximum_path(logp, attn_mask.squeeze(1))
#                     .unsqueeze(1)
#                     .detach()
#                 ) # [b, 1, t_dec, t_enc]
#             z_m = torch.matmul(
#                 attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)
#             ).transpose(
#                 1, 2
#             )  # [b, t_dec, t_enc], [b, t_enc, d] -> [b, d, t_dec]
#             z_logs = torch.matmul(
#                 attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)
#             ).transpose(
#                 1, 2
#             )  # [b, t_dec, t_enc], [b, t_enc, d] -> [b, d, t_dec]
#             logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

#             # Project pitch and energy using attention
#             print(f"Shape of attn before matmul: {attn.shape}") # [b, 1, t_dec, t_enc]
#             print(f"Shape of pitch before unsqueeze: {pitch.shape}") # [b, 1, 345]

#             pred_pitch = torch.matmul(attn.squeeze(1), pitch.squeeze(1).unsqueeze(-1)).unsqueeze(1)
#             pred_energy = torch.matmul(attn.squeeze(1), energy.squeeze(1).unsqueeze(-1)).unsqueeze(1)

#             print(f"Shape of pred_pitch after matmul: {pred_pitch.shape}") # [b, 1, 1222]
#             print(f"Shape of pred_energy after matmul: {pred_energy.shape}")# [b, 1, 1222]

#             return (
#                 (z, z_m, z_logs, logdet, z_mask),
#                 (x_m, x_logs, x_mask),
#                 (attn, logw, logw_),
#                 pred_pitch,
#                 pred_energy,
#             )

#     def preprocess(self, y, y_lengths, y_max_length):
#         if y_max_length is not None:
#             y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
#             y = y[:, :, :y_max_length]
#         y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
#         return y, y_lengths, y_max_length

#     def store_inverse(self):
#         self.decoder.store_inverse()
        

#     def preprocess(self, y, y_lengths, y_max_length):
#         if y_max_length is not None:
#             y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
#             y = y[:, :, :y_max_length]
#         y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
#         return y, y_lengths, y_max_length

#     def store_inverse(self):
#         self.decoder.store_inverse()

import math
import torch
from torch import nn
from torch.nn import functional as F

import modules
import common_modified
import attentions
import monotonic_align

# --- Predictors ---

class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = attentions.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = attentions.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class PitchPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = attentions.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = attentions.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class EnergyPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = attentions.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = attentions.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask

# --- Text Encoder ---

class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        filter_channels_dp,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        window_size=None,
        block_length=None,
        mean_only=False,
        prenet=False,
        gin_channels=0,
    ):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        if prenet:
            self.pre = modules.ConvReluNorm(
                hidden_channels, hidden_channels, hidden_channels,
                kernel_size=5, n_layers=3, p_dropout=0.5,
            )
        else:
            self.pre = None

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
        )

        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        if not mean_only:
            self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
        else:
            self.proj_s = None

        # New predictors
        self.proj_w = DurationPredictor(
            hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout
        )
        self.pitch_predictor = PitchPredictor(
            hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout
        )
        self.energy_predictor = EnergyPredictor(
            hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout
        )
    def forward(self, x, x_lengths, g=None):
            x = self.emb(x) * math.sqrt(self.emb.embedding_dim)  # [b, t, h]
            x = torch.transpose(x, 1, -1)  # [b, h, t]
            x_mask = torch.unsqueeze(common_modified.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

            if self.pre is not None:
                x = self.pre(x, x_mask)
            x = self.encoder(x, x_mask)

            if g is not None:
                g_exp = g.expand(-1, -1, x.size(-1))
                x_dp = torch.cat([torch.detach(x), g_exp], 1)
            else:
                x_dp = torch.detach(x)

            x_m = self.proj_m(x) * x_mask

            if self.proj_s is not None:
                x_logs = self.proj_s(x) * x_mask
            else:
                x_logs = torch.zeros_like(x_m)

            logw = self.proj_w(x_dp, x_mask)
            pitch = self.pitch_predictor(x_dp, x_mask)
            energy = self.energy_predictor(x_dp, x_mask)

            return x_m, x_logs, logw, pitch, energy, x_mask


class FlowSpecDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_blocks,
        n_layers,
        p_dropout=0.0,
        n_split=4,
        n_sqz=2,
        sigmoid_scale=False,
        gin_channels=0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for b in range(n_blocks):
            self.flows.append(modules.ActNorm(channels=in_channels * n_sqz))
            self.flows.append(
                modules.InvConvNear(channels=in_channels * n_sqz, n_split=n_split)
            )
            self.flows.append(
                attentions.CouplingBlock(
                    in_channels * n_sqz,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    gin_channels=gin_channels,
                    p_dropout=p_dropout,
                    sigmoid_scale=sigmoid_scale,
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        if self.n_sqz > 1:
            x, x_mask = common_modified.squeeze(x, x_mask, self.n_sqz)
        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
        if self.n_sqz > 1:
            x, x_mask = common_modified.unsqueeze(x, x_mask, self.n_sqz)
        return x, logdet_tot

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()


class FlowGenerator(nn.Module):
    def __init__(
        self,
        n_vocab,
        hidden_channels,
        filter_channels,
        filter_channels_dp,
        out_channels,
        kernel_size=3,
        n_heads=2,
        n_layers_enc=6,
        p_dropout=0.0,
        n_blocks_dec=12,
        kernel_size_dec=5,
        dilation_rate=5,
        n_block_layers=4,
        p_dropout_dec=0.0,
        n_speakers=0,
        gin_channels=0,
        n_split=4,
        n_sqz=1,
        sigmoid_scale=False,
        window_size=None,
        block_length=None,
        mean_only=False,
        hidden_channels_enc=None,
        hidden_channels_dec=None,
        prenet=False,
        **kwargs
    ):

        super().__init__()
        self.n_vocab = n_vocab
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_heads = n_heads
        self.n_layers_enc = n_layers_enc
        self.p_dropout = p_dropout
        self.n_blocks_dec = n_blocks_dec
        self.kernel_size_dec = kernel_size_dec
        self.dilation_rate = dilation_rate
        self.n_block_layers = n_block_layers
        self.p_dropout_dec = p_dropout_dec
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.window_size = window_size
        self.block_length = block_length
        self.mean_only = mean_only
        self.hidden_channels_enc = hidden_channels_enc
        self.hidden_channels_dec = hidden_channels_dec
        self.prenet = prenet

        self.encoder = TextEncoder(
            n_vocab,
            out_channels,
            hidden_channels_enc or hidden_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_layers_enc,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
            mean_only=mean_only,
            prenet=prenet,
            gin_channels=gin_channels,
        )

        self.decoder = FlowSpecDecoder(
            out_channels,
            hidden_channels_dec or hidden_channels,
            kernel_size_dec,
            dilation_rate,
            n_blocks_dec,
            n_block_layers,
            p_dropout=p_dropout_dec,
            n_split=n_split,
            n_sqz=n_sqz,
            sigmoid_scale=sigmoid_scale,
            gin_channels=gin_channels,
        )

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)


    def forward(
        self,
        x,
        x_lengths,
        y=None,
        y_lengths=None,
        pitch=None,
        energy=None,
        g=None,
        gen=False,
        noise_scale=1.0,
        length_scale=1.0,
    ):
        if g is not None:
            g = F.normalize(self.emb_g(g)).unsqueeze(-1)  # [b, h]
        # x_m, x_logs, logw, x_mask = self.encoder(x, x_lengths, g=g)
        x_m, x_logs, logw, pitch, energy, x_mask = self.encoder(x, x_lengths, g=g)

        print(f"Shape of pitch from encoder: {pitch.shape}")  # [b, 1, t_enc] where t_enc is usually x.size(1)
        print(f"Shape of energy from encoder: {energy.shape}") # [b, 1, t_enc]
        print(f"Shape of x_mask: {x_mask.shape}")      # [b, 1, t_enc]

        if gen:
            w = torch.exp(logw) * x_mask * length_scale
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_max_length = None
        else:
            y_max_length = y.size(2)
        y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
        z_mask = torch.unsqueeze(common_modified.sequence_mask(y_lengths, y_max_length), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

        print(f"Shape of y: {y.shape}")             # [b, n_mel, t_dec]
        print(f"Shape of z_mask: {z_mask.shape}")   # [b, 1, t_dec]
        print(f"Shape of attn_mask: {attn_mask.shape}") # [b, 1, t_dec, t_enc]

        if gen:
            attn = common_modified.generate_path(
                w_ceil.squeeze(1), attn_mask.squeeze(1)
            ).unsqueeze(1)
            z_m = torch.matmul(
                attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)
            ).transpose(
                1, 2
            )  # [b, t_dec, t_enc], [b, t_enc, d] -> [b, d, t_dec]
            z_logs = torch.matmul(
                attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)
            ).transpose(
                1, 2
            )  # [b, t_dec, t_enc], [b, t_enc, d] -> [b, d, t_dec]
            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

            # Project pitch and energy using attention during generation (optional, might need different logic)
            pred_pitch = torch.matmul(attn.squeeze(1), pitch.squeeze(1).unsqueeze(-1)).unsqueeze(1)
            pred_energy = torch.matmul(attn.squeeze(1), energy.squeeze(1).unsqueeze(-1)).unsqueeze(1)

            z = (z_m + torch.exp(z_logs) * torch.randn_like(z_m) * noise_scale) * z_mask
            y, logdet = self.decoder(z, z_mask, g=g, reverse=True)
            return (
                (y, z_m, z_logs, logdet, z_mask),
                (x_m, x_logs, x_mask),
                (attn, logw, logw_),
                pred_pitch,
                pred_energy,
            )
        else:
            z, logdet = self.decoder(y, z_mask, g=g, reverse=False)
            with torch.no_grad():
                x_s_sq_r = torch.exp(-2 * x_logs)
                logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(
                    -1
                )  # [b, t_enc, 1]
                logp2 = torch.matmul(
                    x_s_sq_r.transpose(1, 2), -0.5 * (z ** 2)
                )  # [b, t_enc, d] x [b, d, t_dec] = [b, t_enc, t_dec]
                logp3 = torch.matmul(
                    (x_m * x_s_sq_r).transpose(1, 2), z
                )  # [b, t_enc, d] x [b, d, t_dec] = [b, t_enc, t_dec]
                logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(
                    -1
                )  # [b, t_enc, 1]
                logp = logp1 + logp2 + logp3 + logp4  # [b, t_enc, t_dec]

                attn = (
                    monotonic_align.maximum_path(logp, attn_mask.squeeze(1))
                    .unsqueeze(1)
                    .detach()
                ) # [b, 1, t_dec, t_enc]
            z_m = torch.matmul(
                attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)
            ).transpose(
                1, 2
            )  # [b, t_dec, t_enc], [b, t_enc, d] -> [b, d, t_dec]
            z_logs = torch.matmul(
                attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)
            ).transpose(
                1, 2
            )  # [b, t_dec, t_enc], [b, t_enc, d] -> [b, d, t_dec]
            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

            # Project pitch and energy using attention
            attn_squeezed = attn.squeeze(1)  # [b, t_dec, t_enc]
            pitch_squeezed = pitch.squeeze(1) # [b, t_enc]
            pitch_unsqueezed = pitch_squeezed.unsqueeze(-1) # [b, t_enc, 1]
            pred_pitch = torch.matmul(attn_squeezed.transpose(1, 2), pitch_unsqueezed).unsqueeze(1)

            energy_squeezed = energy.squeeze(1) # [b, t_enc]
            energy_unsqueezed = energy_squeezed.unsqueeze(-1) # [b, t_enc, 1]
            pred_energy = torch.matmul(attn_squeezed.transpose(1, 2), energy_unsqueezed).unsqueeze(1)

            return (
                (z, z_m, z_logs, logdet, z_mask),
                (x_m, x_logs, x_mask),
                (attn, logw, logw_),
                pred_pitch,
                pred_energy,
            )

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
            y = y[:, :, :y_max_length]
        y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
        return y, y_lengths, y_max_length

    def store_inverse(self):
        self.decoder.store_inverse()
