import albumentations as A
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import os
from albumentations.pytorch import ToTensorV2
from PIL import Image
import time
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
    load_checkpoint(torch.load(checkpoint), model, optimizer, lr)
    # reading image
    # image = Image.open(os.path.join(image_path))
    # image = np.array(image)
    image = cv2.imread(os.path.join(image_path))

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
    import time
    start_time = time.time()
    landmarks = model(image)
    end_time = time.time()
    total_time = end_time - start_time
    print("Time: ", total_time)
    landmarks = landmarks.reshape(68, 2)
    print(landmarks)
    image = image[0].permute(1, 2, 0)
    image = image.cpu().detach().numpy()
    vis_keypoints(image_t.astype(np.uint8), list(landmarks.cpu().detach().numpy()), color=KEYPOINT_COLOR,
                  diameter=2)


# test the model ==> print the average RMSE
# def test_model(model, test_loader, loss_fn, device):
#     pass


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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for point in keypoints:
        x, y = point[0], point[1]
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)
    cv2.imshow('image', image)
    cv2.waitKey(0)


# get the mean and the std for each channel of images
def get_mean_std(loader):
    nimages = 0
    mean = 0.0
    var = 0.0
    for i_batch, batch_target in enumerate(loader):
        batch = batch_target[0]
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.cpu().detach().numpy().astype(np.uint8).mean(2).sum(0)
        var += batch.cpu().detach().numpy().astype(np.uint8).var(2).sum(0)
    print("le mean est :", mean)
    print("le std est :", var)
    mean /= nimages
    var /= nimages
    std = torch.sqrt(torch.from_numpy(var))
    return torch.from_numpy(mean), std
    # print(mean)
    # print(std)
