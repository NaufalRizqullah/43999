from torchvision.models import ResNet101_Weights
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def fasterrcnn_backbone_resnet101(num_classes, freeze_model=False):
    TRAINABLE_LAYERS = 5
    
    backbone = resnet_fpn_backbone(backbone_name='resnet101', weights=ResNet101_Weights.DEFAULT, trainable_layers=TRAINABLE_LAYERS)
        
    model = FasterRCNN(backbone, num_classes)

    if freeze_model:
        for param in model.parameters():
            param.requires_grad = False
    
    return model