"""
Part 5 – Baseline 1D CNN Model
===============================
Depends on: part2_label_mapping.py (SUPERCLASSES)

Changes from original:
    - ResidualBlock dropout: 0.2 → 0.3 (stronger regularisation)
    - Head dropout: 0.5 → 0.6
    - Added BatchNorm1d before final Linear in head
    - Added intermediate Linear(256) layer in head for better regularisation

Architecture:
    Input : (batch, 12, 5000)  — 12 leads × 5000 time steps
    Stem  : Conv1d block to project leads into feature space
    Body  : 4 × ResidualBlock (1D) with progressive channel widening
              and stride-2 downsampling to compress time dimension
    Head  : Global Average Pooling → BN → Dropout → Linear(256) → Dropout → Linear(5)
    Output: (batch, 5)  raw logits  (no sigmoid — FocalLoss handles it)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from part2_label_mapping import SUPERCLASSES

NUM_CLASSES  = len(SUPERCLASSES)   # 5
INPUT_LEADS  = 12
INPUT_LENGTH = 5000


# ──────────────────────────────────────────────
# BUILDING BLOCKS
# ──────────────────────────────────────────────
class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation channel attention for 1D signals.
    Recalibrates channel-wise feature responses adaptively.
    """
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x).squeeze(-1)
        scale = self.fc(scale).unsqueeze(-1)
        return x * scale


class ResidualBlock1D(nn.Module):
    """
    1D Residual block with two Conv layers, BN, ReLU, SE attention.

    Changes: dropout increased from 0.2 → 0.3 for stronger regularisation.
    """
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int   = 7,
        stride      : int   = 1,
        dropout     : float = 0.3,   # ← increased from 0.2
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        self.se = SEBlock1D(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        out = self.se(out)
        out = out + self.shortcut(x)
        return self.relu(out)


# ──────────────────────────────────────────────
# BASELINE CNN
# ──────────────────────────────────────────────
class BaselineCNN(nn.Module):
    """
    Baseline 1D CNN for multi-label ECG superclass classification.

    Input  : (B, 12, 5000)
    Output : (B, 5) logits

    Architecture summary:
        Stem  : Conv(12→64, k=15, s=2) → BN → ReLU → MaxPool(3,2)
        Stage1: ResBlock(64→64,  k=7, s=1) × 2
        Stage2: ResBlock(64→128, k=7, s=2) × 2
        Stage3: ResBlock(128→256,k=7, s=2) × 2
        Stage4: ResBlock(256→512,k=7, s=2) × 2
        Head  : GlobalAvgPool → BN → Dropout(0.6) → Linear(256) → Dropout(0.3) → Linear(5)
    """
    def __init__(
        self,
        num_classes  : int   = NUM_CLASSES,
        dropout_head : float = 0.6,   # ← increased from 0.5
    ) -> None:
        super().__init__()

        # ── Stem ─────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(INPUT_LEADS, 64, kernel_size=15, stride=2,
                      padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # ── Residual Stages ───────────────────────────────────────────
        self.stage1 = nn.Sequential(
            ResidualBlock1D(64,  64,  kernel_size=7, stride=1),
            ResidualBlock1D(64,  64,  kernel_size=7, stride=1),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock1D(64,  128, kernel_size=7, stride=2),
            ResidualBlock1D(128, 128, kernel_size=7, stride=1),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock1D(128, 256, kernel_size=7, stride=2),
            ResidualBlock1D(256, 256, kernel_size=7, stride=1),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock1D(256, 512, kernel_size=7, stride=2),
            ResidualBlock1D(512, 512, kernel_size=7, stride=1),
        )

        # ── Classification Head ───────────────────────────────────────
        # Added: BN before dropout, intermediate Linear(256) bottleneck
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),          # (B, 512, 1)
            nn.Flatten(),                      # (B, 512)
            nn.BatchNorm1d(512),              # ← NEW: normalise before dropout
            nn.Dropout(dropout_head),          # ← 0.6
            nn.Linear(512, 256),              # ← NEW: bottleneck layer
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_head * 0.5),   # ← 0.3 lighter second dropout
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """Return intermediate feature maps for Grad-CAM in Part 9."""
        maps = {}
        x = self.stem(x);   maps["stem"]   = x
        x = self.stage1(x); maps["stage1"] = x
        x = self.stage2(x); maps["stage2"] = x
        x = self.stage3(x); maps["stage3"] = x
        x = self.stage4(x); maps["stage4"] = x
        return maps


# ──────────────────────────────────────────────
# MODEL FACTORY
# ──────────────────────────────────────────────
def build_baseline_cnn(device: str = "cpu") -> BaselineCNN:
    model = BaselineCNN(num_classes=NUM_CLASSES)
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    model = build_baseline_cnn(device)
    print(f"[INFO] Trainable parameters: {count_parameters(model):,}")

    print("\n── Model Summary ────────────────────────────────────────────")
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
    assert logits.shape == (4, NUM_CLASSES), "Output shape mismatch!"

    maps = model.get_feature_maps(dummy)
    for k, v in maps.items():
        print(f"  {k:<8}: {tuple(v.shape)}")

    print("\n[INFO] ✓ Baseline CNN smoke test passed.")