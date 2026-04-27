from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_model() -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1)
    )
    return model


if __name__ == "__main__":
    build_model()
