from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_model() -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(2048, 1)
    return model


if __name__ == "__main__":
    build_model()
