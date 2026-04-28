import timm
from torch import nn


def get_backbone():
    return timm.create_model("efficientnet_b0", num_classes=0, pretrained=True)


def build_model(head_inputs: int = 256, dropout_prob: float = 0.5) -> nn.Sequential:
    backbone = get_backbone()
    for p in backbone.parameters():
        p.requires_grad = False

    head = nn.Sequential(
        nn.Linear(backbone.num_features, head_inputs),
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Linear(head_inputs, 1),
    )
    model = nn.Sequential(backbone, head)
    return model


if __name__ == "__main__":
    model = build_model()
    for name, p in model.named_parameters():
        print(name, p.requires_grad)
