import cv2
import numpy as np
from fatigueDetector import *
import torch
from efficientnet_pytorch import EfficientNet
from torch import nn, optim
from run import CHECKPOINT_FILE, DEVICE, pretrained_model_name
import Config
from Utils import load_checkpoint, vis_keypoints


def captureFace() -> bool:
    """Function that capture live photo out of the camera and run the model for key points output
    """
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 5e-5
    model = EfficientNet.from_pretrained(pretrained_model_name)
    model._fc = nn.Linear(in_features=1280, out_features=68 * 2, bias=True)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    load_checkpoint(torch.load(CHECKPOINT_FILE, map_location=torch.device(DEVICE)), model, optimizer, LEARNING_RATE)
    #
    capture = cv2.VideoCapture(0)
    while True:
        has_frame, frame = capture.read()
        if not has_frame:
            print('Can\'t get frame')
            break
        frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
        image_t = frame
        frame = Config.transforms(image=frame)["image"]
        frame = frame.reshape(1, 3, 512, 512)
        frame = frame.to(device='cpu', dtype=torch.float32)
        landmarks = model(frame)
        landmarks = landmarks.reshape(68, 2)
        # print("=== EYES CLOSED STATE ===")
        value1, firstEyeClosed1 = isEyeClosed(landmarks[38:44, :].cpu().detach().numpy())
        value2, secondEyeClosed2 = isEyeClosed(landmarks[44:50, :].cpu().detach().numpy())
        # print("=== MOUTH OPEN STATE ===")
        # mouth key points 48..66 => 49..68
        value3, mouthOpen = isYawning(landmarks[49:68, :].cpu().detach().numpy())
        frame = frame[0].permute(1, 2, 0)
        frame = frame.cpu().detach().numpy()
        for point in landmarks:
            x, y = point[0], point[1]
            cv2.circle(image_t, (int(x), int(y)), 2, (0, 255, 0), -1)
        text = "EYES: " + ("CLOSED" if (firstEyeClosed1 and secondEyeClosed2) else "OPEN") + "       " + (
            "MOUTH: " + "OPEN" if mouthOpen else "CLOSED")
        cv2.putText(image_t, text, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', image_t)
        key = cv2.waitKey(3)
        if key == 27:
            print('Pressed Esc')
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    captureFace()
