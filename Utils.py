import albumentations as A
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import os
from albumentations.pytorch import ToTensorV2
import Config


def split_train_val_test(images, labels, train=0.6, test=0.2, val=0.2):
    train_index: int = int(images.shape[0] * train)
    test_index: int = train_index + int(images.shape[0] * test)
    val_index: int = test_index + train_index + int(images.shape[0] * val)
    print(labels[:train_index, ].reshape(labels[:train_index, ].shape[0], 68 * 2).shape)
    # [X_train,X_test,X_val,y_train,y_test,y_val]
    return [images[:train_index, ],
            images[train_index: test_index, ],
            images[test_index: val_index, ],
            labels[:train_index, ],
            labels[train_index: test_index, ],
            labels[test_index: val_index, ]]


def plot_image_with_landmarks(checkpoint, model, optimizer, device, lr, image_path):
    image = cv2.imread(image_path)
    image = Config.transforms(image=image)["image"]
    image = image.reshape(1, 3, 512, 512)
    image = image.to(device=device, dtype=torch.float32)
    load_checkpoint(torch.load(checkpoint), model, optimizer, lr)
    landmarks = model(image)
    landmarks = landmarks.reshape(68, 2)
    image = image[0].permute(1, 2, 0)
    image = image.cpu().detach().numpy()
    vis_keypoints(image.astype(np.uint8), list(landmarks.cpu().detach().numpy()), color=KEYPOINT_COLOR,
                  diameter=2)


# test the model ==> print the average RMSE
def test_model(model, test_loader, loss_fn, device):
    losses = []
    loop = tqdm(test_loader)
    num_examples = 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loop):
            data = X.to(device=device, dtype=torch.float32)
            targets = y.to(device=device, dtype=torch.float32)
            scores = model(data)
            loss = loss_fn(scores, targets)
            num_examples += torch.numel(scores[targets != -1])
            losses.append(loss.item())
    print(f"Loss average in the test: {(sum(losses) / num_examples) ** 0.5}")


def get_images_labels(all_data_path):
    f = open(all_data_path, )
    data = json.load(f)
    L_images = []
    L_labels = []
    for item in data:
        L_images.append(data[item]["file_name"])
        L_labels.append(data[item]["face_landmarks"])
    f.close()

    return np.array(L_images), np.array(L_labels)


KEYPOINT_COLOR = (0, 255, 0)  # Green


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=2):
    image = image.copy()
    for point in keypoints:
        x, y = point[0], point[1]
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    cv2.imshow('image', image)
    cv2.waitKey(0)


# get the mean and the std for each channel of images
def get_mean_std(loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    return mean, std
