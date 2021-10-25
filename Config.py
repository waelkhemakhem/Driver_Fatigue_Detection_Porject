import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-5
BATCH_SIZE = 4
NUM_EPOCHS = 12
CHECKPOINT_FILE = "KeyPointsModel.pth.tar"
PIN_MEMORY = True
NUM_WORKERS = 0
SAVE_MODEL = True
LOAD_MODEL = False

# Data augmentation for images
train_transforms = A.Compose(
    [
        A.Resize(width=512, height=512),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        # A.geometric.transforms.Affine(shear=20, scale=1.0, mode="constant", p=0.2),
        A.RandomBrightnessContrast(contrast_limit=0.5, brightness_limit=0.5, p=0.2),
        A.OneOf([
            A.GaussNoise(p=0.8),
            A.CLAHE(p=0.8),
            A.ImageCompression(p=0.8),
            A.RandomGamma(p=0.8),
            A.Posterize(p=0.8),
            A.Blur(p=0.8),
        ], p=1.0),
        # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(
            mean=[131.6930, 109.5418, 98.3958],
            std=[66.2921, 60.1082, 59.9244],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
)

val_transforms = A.Compose(
    [
        A.Resize(height=512, width=512),
        A.Normalize(
            mean=[131.6930, 109.5418,  98.3958],
            std=[66.2921, 60.1082, 59.9244],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False), )

transforms = A.Compose(
    [
        A.Resize(height=512, width=512),
        A.Normalize(
            mean=[131.6930, 109.5418, 98.3958],
            std=[66.2921, 60.1082, 59.9244],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

# invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
#                                                     std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
#                                transforms.Normalize(mean=[-0.485, -0.456, -0.406],
#                                                     std=[1., 1., 1.]),
#                                ])
#
# inv_tensor = invTrans(inp_tensor)
