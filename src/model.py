"""STFPM model: a frozen teacher, a trainable student, and their discrepancy.

Student-Teacher Feature-Pyramid Matching (Wang et al., BMVC 2021). A pretrained
ImageNet backbone (the *teacher*) is frozen; a second backbone of the same
architecture (the *student*) is trained from scratch to reproduce the teacher's
feature maps on *normal* images. At test time, wherever the student fails to
match the teacher — measured across several feature-pyramid levels — is flagged
as anomalous.

This module holds the network and two pure functions: `distillation_loss` (the
training objective) and `anomaly_map` (the per-pixel score used at inference).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

# Backbone constructors paired with their default ImageNet weights.
_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "wide_resnet50_2": (
        models.wide_resnet50_2,
        models.Wide_ResNet50_2_Weights.IMAGENET1K_V1,
    ),
}

# ResNet stages in depth order; a subset of these are the pyramid taps.
_STAGES = ("layer1", "layer2", "layer3", "layer4")


class FeatureExtractor(nn.Module):
    """A ResNet backbone that returns feature maps at selected pyramid stages.

    Runs the stem and residual stages in order and collects the outputs of the
    requested stages, stopping once the deepest requested stage is reached so no
    unused layers are computed.

    Attributes:
        layers (tuple[str, ...]): Stage names whose outputs are returned, e.g.
            ("layer1", "layer2", "layer3").
    """

    def __init__(self, backbone: str, layers: tuple[str, ...], pretrained: bool):
        """Build the extractor from a torchvision backbone.

        Args:
            backbone (str): Backbone key, one of `_BACKBONES`.
            layers (tuple[str, ...]): Stage names to tap, a subset of `_STAGES`.
            pretrained (bool): If True, load ImageNet weights (the teacher); if
                False, randomly initialise (the student).

        Raises:
            KeyError: If `backbone` is not a known backbone.
        """
        super().__init__()
        ctor, weights = _BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)
        self.layers = tuple(layers)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleDict({name: getattr(net, name) for name in _STAGES})

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract feature maps at the configured stages.

        Args:
            x (torch.Tensor): Input batch of shape (N, 3, H, W).

        Returns:
            list[torch.Tensor]: One feature map per requested stage, in depth
                order, each of shape (N, C_l, H_l, W_l).
        """
        x = self.stem(x)
        feats: list[torch.Tensor] = []
        for name in _STAGES:
            x = self.stages[name](x)
            if name in self.layers:
                feats.append(x)
            if len(feats) == len(self.layers):
                break
        return feats


class STFPM(nn.Module):
    """A frozen teacher paired with a trainable student for feature matching.

    Attributes:
        teacher (FeatureExtractor): Frozen, pretrained backbone; never trained.
        student (FeatureExtractor): Randomly initialised backbone; the only part
            with gradients.
    """

    def __init__(self, backbone: str, layers: tuple[str, ...]):
        """Build the teacher/student pair.

        Args:
            backbone (str): Backbone key shared by teacher and student.
            layers (tuple[str, ...]): Feature-pyramid stages to compare.
        """
        super().__init__()
        self.teacher = FeatureExtractor(backbone, layers, pretrained=True)
        self.student = FeatureExtractor(backbone, layers, pretrained=False)
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.teacher.eval()

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Extract teacher and student features for the same input.

        Args:
            x (torch.Tensor): Input batch of shape (N, 3, H, W).

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: The teacher feature
                maps (detached) and the student feature maps, aligned by stage.
        """
        with torch.no_grad():
            teacher_feats = self.teacher(x)
        student_feats = self.student(x)
        return teacher_feats, student_feats

    def train(self, mode: bool = True) -> "STFPM":
        """Set training mode while keeping the teacher frozen in eval mode.

        Args:
            mode (bool): Whether to put the student in training mode.

        Returns:
            STFPM: This module, for chaining.
        """
        super().train(mode)
        self.teacher.eval()  # frozen batch-norm stats: teacher is always eval
        return self


def distillation_loss(
    teacher_feats: list[torch.Tensor], student_feats: list[torch.Tensor]
) -> torch.Tensor:
    """Compute the student-teacher feature-matching loss, summed over stages.

    Each feature map is L2-normalised along the channel dimension so the match
    is directional (magnitude-invariant), then compared with mean squared error.

    Args:
        teacher_feats (list[torch.Tensor]): Teacher feature maps, one per stage.
        student_feats (list[torch.Tensor]): Student feature maps, aligned by
            stage with `teacher_feats`.

    Returns:
        torch.Tensor: Scalar loss (sum of per-stage MSE on normalised features).
    """
    loss = teacher_feats[0].new_zeros(())
    for teacher, student in zip(teacher_feats, student_feats):
        teacher = F.normalize(teacher, dim=1)
        student = F.normalize(student, dim=1)
        loss = loss + F.mse_loss(student, teacher)
    return loss


def anomaly_map(
    teacher_feats: list[torch.Tensor],
    student_feats: list[torch.Tensor],
    out_size: tuple[int, int],
) -> torch.Tensor:
    """Build a per-pixel anomaly map by fusing per-stage discrepancies.

    For each stage, the squared L2 distance between the channel-normalised
    teacher and student features gives a coarse discrepancy map; each is
    upsampled to `out_size` and the stages are multiplied together, so a pixel
    scores high only where several scales agree it is anomalous.

    Args:
        teacher_feats (list[torch.Tensor]): Teacher feature maps, one per stage.
        student_feats (list[torch.Tensor]): Student feature maps, aligned by
            stage with `teacher_feats`.
        out_size (tuple[int, int]): Target (H, W) for the returned map.

    Returns:
        torch.Tensor: Anomaly map of shape (N, 1, H, W); higher is more
            anomalous.
    """
    amap = torch.ones(
        teacher_feats[0].shape[0], 1, *out_size, device=teacher_feats[0].device
    )
    for teacher, student in zip(teacher_feats, student_feats):
        teacher = F.normalize(teacher, dim=1)
        student = F.normalize(student, dim=1)
        dist = 0.5 * torch.sum((teacher - student) ** 2, dim=1, keepdim=True)
        dist = F.interpolate(dist, size=out_size, mode="bilinear", align_corners=False)
        amap = amap * dist
    return amap
