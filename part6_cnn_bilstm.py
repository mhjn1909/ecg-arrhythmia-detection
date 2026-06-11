"""
Part 6 – Improved CNN + BiLSTM Hybrid Model
============================================
Depends on: part2_label_mapping.py (SUPERCLASSES)
            part5_baseline_cnn.py  (SEBlock1D, ResidualBlock1D)

Changes from original:
    - dropout_lstm: 0.3 → 0.4
    - dropout_head: 0.5 → 0.6
    - Added VariationalDropout wrapper for LSTM
      (drops same units across all timesteps — more effective than standard dropout for RNNs)
    - Added BN after dual pooling concat in head
    - Intermediate Linear bottleneck: 1024 → 512 → 256 → 5 (was 1024 → 256 → 5)
    - Added extra Dropout(0.3) between bottleneck layers

Architecture:
    Input  : (B, 12, 5000)
    Stage 1: Multi-scale CNN stem (k=7,15,31 parallel branches)
    Stage 2: 4 × ResidualBlock1D with SE attention
    Stage 3: BiLSTM (2 layers) + VariationalDropout
    Stage 4: Multi-head Self-Attention
    Head   : Concat(AvgPool, MaxPool) → BN → Dropout → Linear(512) → Dropout → Linear(256) → Dropout → Linear(5)
    Output : (B, 5) raw logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from part2_label_mapping import SUPERCLASSES
from part5_baseline_cnn  import SEBlock1D, ResidualBlock1D

NUM_CLASSES  = len(SUPERCLASSES)
INPUT_LEADS  = 12
INPUT_LENGTH = 5000


# ══════════════════════════════════════════════════════════════════
# VARIATIONAL DROPOUT FOR LSTM
# ══════════════════════════════════════════════════════════════════
class VariationalDropout(nn.Module):
    """
    Variational (locked) dropout for RNN sequences.

    Standard dropout drops different neurons at each timestep.
    Variational dropout samples ONE mask per forward pass and applies
    it identically across all timesteps — this is much more effective
    at regularising RNNs because the model cannot simply average out
    the noise over time.

    Reference: Gal & Ghahramani, "A Theoretically Grounded Application
               of Dropout in Recurrent Neural Networks" (NeurIPS 2016)

    Args:
        p : dropout probability
    """
    def __init__(self, p: float = 0.4) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, C) — LSTM output sequence
        Returns:
            Dropped tensor of same shape
        """
        if not self.training or self.p == 0:
            return x
        # Sample mask on (B, 1, C) — same mask for every timestep
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        mask = mask / (1 - self.p)   # scale to maintain expected value
        return x * mask              # broadcast across T


# ══════════════════════════════════════════════════════════════════
# MULTI-SCALE STEM (unchanged)
# ══════════════════════════════════════════════════════════════════
class MultiScaleStem(nn.Module):
    """
    Parallel convolutions at three kernel sizes:
        k=7  → 14 ms  (QRS spike)
        k=15 → 30 ms  (QRS morphology)
        k=31 → 62 ms  (P-wave, T-wave, ST segment)
    """
    def __init__(self, in_channels: int = 12, out_channels: int = 96) -> None:
        super().__init__()
        branch_ch = out_channels // 3

        def _branch(k: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(in_channels, branch_ch, kernel_size=k,
                          stride=2, padding=k // 2, bias=False),
                nn.BatchNorm1d(branch_ch),
                nn.ReLU(inplace=True),
            )

        self.branch_fine   = _branch(7)
        self.branch_medium = _branch(15)
        self.branch_coarse = _branch(31)

        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f   = self.branch_fine(x)
        m   = self.branch_medium(x)
        c   = self.branch_coarse(x)
        out = torch.cat([f, m, c], dim=1)
        return self.fuse(out)


# ══════════════════════════════════════════════════════════════════
# TEMPORAL ATTENTION (unchanged)
# ══════════════════════════════════════════════════════════════════
class TemporalAttention(nn.Module):
    """
    Lightweight multi-head self-attention over the time dimension.
    Stores attention weights for Part 9 XAI.
    """
    def __init__(self, d_model: int, n_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.attn = nn.MultiheadAttention(
            embed_dim    = d_model,
            num_heads    = n_heads,
            dropout      = dropout,
            batch_first  = True,
        )
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights: torch.Tensor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, weights = self.attn(x, x, x, need_weights=True,
                                      average_attn_weights=True)
        self.last_attn_weights = weights.detach()
        return self.norm(x + self.dropout(attn_out))


# ══════════════════════════════════════════════════════════════════
# CNN + BiLSTM MODEL
# ══════════════════════════════════════════════════════════════════
class CNNBiLSTM(nn.Module):
    """
    Improved ECG classifier: MultiScaleStem → ResidualCNN →
    BiLSTM + VariationalDropout → TemporalAttention → Head

    Key regularisation improvements vs original:
        - VariationalDropout on LSTM output (locked per timestep)
        - Higher dropout throughout (lstm: 0.4, head: 0.6)
        - Deeper head with staged dropout: 1024→512→256→5
        - BN after pooling concat
    """
    def __init__(
        self,
        num_classes   : int   = NUM_CLASSES,
        lstm_hidden   : int   = 256,
        lstm_layers   : int   = 2,
        attn_heads    : int   = 8,
        dropout_lstm  : float = 0.4,   # ← increased from 0.3
        dropout_head  : float = 0.6,   # ← increased from 0.5
    ) -> None:
        super().__init__()

        # ── 1. Multi-scale CNN stem ────────────────────────────────────
        self.stem = MultiScaleStem(in_channels=INPUT_LEADS, out_channels=96)

        # ── 2. Residual CNN body ───────────────────────────────────────
        self.cnn_body = nn.Sequential(
            ResidualBlock1D(96,  128, kernel_size=7, stride=1),
            ResidualBlock1D(128, 128, kernel_size=7, stride=2),
            ResidualBlock1D(128, 256, kernel_size=7, stride=2),
            ResidualBlock1D(256, 256, kernel_size=7, stride=2),
        )
        cnn_out_channels = 256

        # ── 3. BiLSTM ─────────────────────────────────────────────────
        self.bilstm = nn.LSTM(
            input_size    = cnn_out_channels,
            hidden_size   = lstm_hidden,
            num_layers    = lstm_layers,
            batch_first   = True,
            bidirectional = True,
            dropout = dropout_lstm if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2   # 512

        # ── NEW: Variational dropout after LSTM ────────────────────────
        self.var_dropout = VariationalDropout(p=dropout_lstm)

        # ── 4. Temporal self-attention ────────────────────────────────
        self.attention = TemporalAttention(
            d_model = lstm_out_dim,
            n_heads = attn_heads,
            dropout = 0.1,
        )

        # ── 5. Classification head ────────────────────────────────────
        # Dual pooling: avg + max → concat → 1024
        # Deeper staged head with BN and multiple dropout layers
        self.head = nn.Sequential(
            nn.BatchNorm1d(lstm_out_dim * 2),      # ← NEW: BN on pooled features
            nn.Dropout(dropout_head),               # ← 0.6
            nn.Linear(lstm_out_dim * 2, 512),       # 1024 → 512
            nn.GELU(),
            nn.Dropout(dropout_head * 0.5),         # ← 0.3
            nn.Linear(512, 256),                    # ← NEW: extra bottleneck
            nn.GELU(),
            nn.Dropout(dropout_head * 0.33),        # ← 0.2
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)
                        n = param.size(0)
                        param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN
        x = self.stem(x)           # (B, 96,  1250)
        x = self.cnn_body(x)       # (B, 256,  157)

        # LSTM
        x = x.permute(0, 2, 1)    # (B, 157, 256)
        x, _ = self.bilstm(x)     # (B, 157, 512)

        # Variational dropout (train only, locked across timesteps)
        x = self.var_dropout(x)    # (B, 157, 512)

        # Self-attention
        x = self.attention(x)      # (B, 157, 512)

        # Dual pooling
        x_avg = x.mean(dim=1)             # (B, 512)
        x_max = x.max(dim=1).values       # (B, 512)
        x     = torch.cat([x_avg, x_max], dim=1)  # (B, 1024)

        return self.head(x)        # (B, 5)

    def get_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return CNN feature maps for Grad-CAM. Shape: (B, 256, 157)"""
        x = self.stem(x)
        return self.cnn_body(x)

    def get_attention_weights(self) -> torch.Tensor:
        """Return last attention weights for XAI. Shape: (B, T, T)"""
        return self.attention.last_attn_weights

    def get_lstm_output(self, x: torch.Tensor) -> torch.Tensor:
        """Return BiLSTM output for saliency analysis. Shape: (B, T, 512)"""
        x = self.stem(x)
        x = self.cnn_body(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        return x


# ══════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ══════════════════════════════════════════════════════════════════
def build_cnn_bilstm(device: str = "cpu") -> CNNBiLSTM:
    model = CNNBiLSTM(num_classes=NUM_CLASSES)
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    model = build_cnn_bilstm(device)
    print(f"[INFO] Trainable parameters: {count_parameters(model):,}")

    print("\n── Model Summary (CNN+BiLSTM) ───────────────────────────────")
    summary(
        model,
        input_size=(2, INPUT_LEADS, INPUT_LENGTH),
        col_names=["input_size", "output_size", "num_params"],
        device=device,
    )

    dummy  = torch.randn(4, INPUT_LEADS, INPUT_LENGTH, device=device)
    logits = model(dummy)
    print(f"\n[INFO] Input  shape : {tuple(dummy.shape)}")
    print(f"[INFO] Output shape : {tuple(logits.shape)}")
    assert logits.shape == (4, NUM_CLASSES)

    cnn_feats = model.get_cnn_features(dummy)
    print(f"[INFO] CNN features  : {tuple(cnn_feats.shape)}")

    _ = model(dummy)
    attn = model.get_attention_weights()
    print(f"[INFO] Attn weights  : {tuple(attn.shape)}")

    print("\n[INFO] ✓ CNN+BiLSTM smoke test passed.")