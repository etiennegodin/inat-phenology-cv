from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_model(head_inputs: int = 256, dropout_prob: float = 0.5) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(in_features, head_inputs),
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Linear(head_inputs, 1),
    )
    return model


if __name__ == "__main__":
    build_model()
