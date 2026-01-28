import torch
import torch.nn as nn
import torchvision.models as models
import timm


def get_resnet50(num_classes, pretrained=True):
    """
    Standard ResNet50 with a custom head for RF Classification.
    """
    print(">> Loading ResNet50...")
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)

    # Freeze backbone if needed (optional, usually we fine-tune)
    # for param in model.parameters():
    #     param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
    return model


def get_vit(num_classes, pretrained=True):
    """
    Vision Transformer (ViT-Base).
    """
    print(">> Loading ViT (vit_base_patch16_224)...")
    model = timm.create_model('vit_base_patch16_224',
                              pretrained=pretrained, num_classes=num_classes)
    return model


def get_mobilenet_v3(num_classes, pretrained=True):
    """
    Small Parameter Baseline: MobileNetV3-Small.
    This is the new addition for comparison against Green-Mamba.
    """
    print(">> Loading MobileNetV3-Small...")
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # Replace the classifier head
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    return model
