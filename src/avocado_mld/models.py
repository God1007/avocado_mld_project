from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class MLDModel(nn.Module):
    def __init__(self, num_stages: int = 5):
        super().__init__()
        backbone = resnet18(weights=None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.latent_head = nn.Linear(feature_dim, 1)
        self.ordinal_thresholds = nn.Parameter(torch.linspace(-1.0, 1.0, num_stages - 1))
        self.ordinal_scale = nn.Parameter(torch.tensor(5.0))
        self.remaining_bias = nn.Parameter(torch.tensor(10.0))
        self.remaining_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        latent = self.latent_head(features).squeeze(1)
        scale = torch.nn.functional.softplus(self.ordinal_scale)
        ordinal_logits = scale * (latent.unsqueeze(1) - self.ordinal_thresholds.unsqueeze(0))
        remaining_days = self.remaining_bias - torch.nn.functional.softplus(self.remaining_scale) * latent
        return {
            "latent": latent,
            "ordinal_logits": ordinal_logits,
            "remaining_days": remaining_days,
        }


class StageClassifier(nn.Module):
    def __init__(self, num_stages: int = 5):
        super().__init__()
        backbone = resnet18(weights=None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_stages)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.classifier(features)
