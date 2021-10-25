import cv2
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image


class Facial_Key_Points(Dataset):
    def __init__(self, images_path, images_name, labels, transform=None):
        super().__init__()
        self.images_path = images_path
        self.images, self.labels = images_name, labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        # image = cv2.imread(os.path.join(self.images_path, image_path))
        image = Image.open(os.path.join(self.images_path, image_path))
        image = np.array(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = self.labels[index]
        if self.transform:
            transformed = self.transform(image=image, keypoints=keypoints)
            image = transformed['image']
            keypoints = transformed['keypoints']
        return image, np.array(keypoints).reshape(68*2,)

