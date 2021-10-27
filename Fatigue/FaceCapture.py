import cv2
import numpy as np
from fatigueDetector import *
import torch
from efficientnet_pytorch import EfficientNet
from torch import nn, optim

import Config
from Utils import load_checkpoint, vis_keypoints


def captureFace():
    # load model
    DEVICE = 'cpu'
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 5e-5
    model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = nn.Linear(in_features=1280, out_features=68 * 2, bias=True)
    model = model.to(DEVICE)
    CHECKPOINT_FILE = "D:/ENSI/3eme/Aprentissage_supervisé/Driver_fatigue_detection_project/models/efficientnet-b0_v2-KeyPointsModel.pth.tar"
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    load_checkpoint(torch.load(CHECKPOINT_FILE), model, optimizer, LEARNING_RATE)
    #
    capture = cv2.VideoCapture(0)
    while True:
        has_frame, image = capture.read()
        if not has_frame:
            print('Can\'t get frame')
            break
            #
        # let's resize our image to be 150 pixels wide, but in order to
        # prevent our resized image from being skewed/distorted, we must
        # first calculate the ratio of the new width to the old width
        r = 512.0 / image.shape[1]
        dim = (512, 512)
        # perform the actual resizing of the image
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        image_t = image
        image = Config.transforms(image=image)["image"]
        image = image.reshape(1, 3, 512, 512)
        image = image.to(device='cpu', dtype=torch.float32)
        landmarks = model(image)
        landmarks = landmarks.reshape(68, 2)
        # print("=== EYES CLOSED STATE ===")
        val1: bool = isEyeClosed(landmarks[38:44, :].cpu().detach().numpy())
        val2: bool = isEyeClosed(landmarks[44:50, :].cpu().detach().numpy())
        # print(val1, val2)
        # print("=== MOUTH OPEN STATE ===")
        # mouth key points 48..66 => 49..68
        val3: bool = isYawning(landmarks[49:68, :].cpu().detach().numpy())
        # print(val3)
        image = image[0].permute(1, 2, 0)
        image = image.cpu().detach().numpy()
        # cv2.circle(image_t, (int(100), int(100)), 2, (0, 255, 0), -1)
        for point in landmarks:
            x, y = point[0], point[1]
            cv2.circle(image_t, (int(x), int(y)), 2, (0, 255, 0), -1)
        # vis_keypoints(image_t.astype(np.uint8), list(landmarks.cpu().detach().numpy()), color=(0, 255, 0),
        #               diameter=2)
        text = "EYES: " + ("CLOSED" if (val1[1] and val2[1]) else "OPEN")+ "       " + (
            "MOUTH: " + "OPEN" if val3[1] else "CLOSED")
        cv2.putText(image_t, text, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', image_t)
        key = cv2.waitKey(3)
        if key == 27:
            print('Pressed Esc')
            break

    capture.release()
    cv2.destroyAllWindows()


captureFace()
