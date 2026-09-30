"""Reference architectures for the null population and the fixtures.

Two families, deliberately. A claim that holds only on one architecture will be
dismissed, and clause 2.2.6's model-agnostic requirement is about exactly this. The
null is indexed by ``(family x corpus x converter)``, so these are the families the
operational and contrast arms are fitted over:

* **ResNet-18** -- the operational arm, on GTSRB. 11.7M parameters, trains in minutes
  on an 8 GB card, which is what makes a 64-model null affordable at all.
* **SmallCNN** -- the family-contrast arm. Deliberately unlike ResNet-18, so
  ``null_family_transfer_delta`` measures an architecture difference rather than a
  depth difference.

Nothing here is downloaded. ``pretrained=True`` is a clause 2.2.6 violation *and* a
reference-contamination bug at the same time, so every weight in this system is either
trained in-house from a declared public corpus and entered in the offline manifest, or
delivered by a vendor and digested on arrival.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SmallCNN", "ResNet18", "build", "FAMILIES"]

FAMILIES = ("resnet18", "smallcnn")


class SmallCNN(nn.Module):
    """A compact convolutional classifier -- the family-contrast arm.

    Kept scriptable: every fixture in ``conformance/models`` is produced by
    ``torch.jit.script`` on this or :class:`ResNet18`, because a TorchScript loader
    that has never been handed a real TorchScript file is untested.
    """

    def __init__(self, n_classes: int = 43, in_channels: int = 3) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block2(self.block1(x))
        return self.classifier(torch.flatten(self.pool(x), 1))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate representation -- what the data-side detectors score in."""
        x = self.block2(self.block1(x))
        return torch.flatten(self.pool(x), 1)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.downsample(x))


class ResNet18(nn.Module):
    """ResNet-18 adapted for small images -- the operational-null architecture.

    3x3 stem and no max-pool, which is the standard CIFAR/GTSRB adaptation. Written out
    rather than imported from torchvision so that the graph is fixed by this file:
    a null fitted under one definition is not valid under another, and a silent
    upstream change to a stem would invalidate 21 GPU-hours without any error.
    """

    def __init__(self, n_classes: int = 43, in_channels: int = 3) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, n_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.classifier(torch.flatten(self.pool(x), 1))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.pool(x), 1)


def build(family: str, n_classes: int = 43, in_channels: int = 3) -> nn.Module:
    """Construct a model by family name.

    Raises on an unknown family rather than falling back to a default. A silent
    fallback would mean a report naming ``null_family: resnet18`` for a model that is
    not one.
    """
    if family == "resnet18":
        return ResNet18(n_classes=n_classes, in_channels=in_channels)
    if family == "smallcnn":
        return SmallCNN(n_classes=n_classes, in_channels=in_channels)
    raise ValueError(f"unknown family {family!r}; known families are {FAMILIES}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
