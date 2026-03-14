import torch.nn as nn
import torchvision.models as models


def build_model(num_classes=5, freeze_backbone=False):
    """
    EfficientNet-B0 fine-tuned for multi-label chest pathology detection.
    Using transfer learning from ImageNet — standard practice in medical imaging
    where labeled data is scarce.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier head for 5-class multi-label output
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model