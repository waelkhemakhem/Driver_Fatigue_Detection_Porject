import torch
from Dataset import Facial_Key_Points
from torch import nn, optim
import os
import Config
from torch.utils.data import DataLoader
from tqdm import tqdm
from Utils import get_images_labels, split_train_val_test, test_model
from torchvision.models import vgg16
from efficientnet_pytorch import EfficientNet
from Utils import (
    load_checkpoint,
    save_checkpoint
)


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    losses = []
    loop = tqdm(loader)
    num_examples = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)

        # forward
        scores = model(data)
        loss = loss_fn(scores, targets)
        num_examples += torch.numel(scores[targets != -1])
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss average over epoch: {(sum(losses) / num_examples) ** 0.5}")


def Create_train_test_val_loader(
        BATCH_SIZE=4,
        PIN_MEMORY=True,
        NUM_WORKERS=0,
        images_path="C:/Users/Utilisateur/Desktop/images/",
        all_data_path="C:/Users/Utilisateur/Desktop/all_data.json",
        train=0.6,
        test=0.2,
        val=0.2
):
    images, labels = get_images_labels(all_data_path)
    X_train, X_test, X_val, y_train, y_test, y_val = split_train_val_test(images, labels, train=train, test=test,
                                                                          val=val)
    train_ds = Facial_Key_Points(
        images_name=X_train,
        labels=y_train,
        images_path=images_path,
        transform=Config.train_transforms
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )
    val_ds = Facial_Key_Points(
        transform=Config.val_transforms,
        images_name=X_val,
        labels=y_val,
        images_path=images_path,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    test_ds = Facial_Key_Points(
        images_name=X_test,
        labels=y_test,
        images_path=images_path,
        transform=Config.val_transforms
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )
    return [train_loader, val_loader, test_loader]


def test(images_path,
         all_data_path,
         CHECKPOINT_FILE,
         model,
         optimizer,
         LEARNING_RATE,
         DEVICE="cuda" if torch.cuda.is_available() else 'cpu'):
    test_loader = Create_train_test_val_loader(
        images_path=images_path,
        all_data_path=all_data_path,
    )[2]
    load_checkpoint(torch.load(CHECKPOINT_FILE), model, optimizer, LEARNING_RATE)
    test_model(model, test_loader, nn.MSELoss(reduction="mean"), DEVICE)


def train(
        DEVICE="cuda" if torch.cuda.is_available() else 'cpu',
        LEARNING_RATE=1e-4,
        WEIGHT_DECAY=5e-5,
        NUM_EPOCHS=12,
        BATCH_SIZE=4,
        PIN_MEMORY=True,
        NUM_WORKERS=0,
        images_path="C:/Users/Utilisateur/Desktop/images/",
        all_data_path="C:/Users/Utilisateur/Desktop/all_data.json",
        CHECKPOINT_FILE="KeyPointsModel.pth.tar",
        SAVE_MODEL=True,
        LOAD_MODEL=False,
        train=0.6,
        test=0.2,
        val=0.2
):
    train_loader = Create_train_test_val_loader(
        BATCH_SIZE,
        PIN_MEMORY,
        NUM_WORKERS,
        images_path,
        all_data_path,
        train=train,
        test=test,
        val=val,
    )[0]
    loss_fn = nn.MSELoss(reduction="mean")
    # pretrained model
    model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = nn.Linear(in_features=1280, out_features=68 * 2, bias=True)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()
    if LOAD_MODEL and CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(CHECKPOINT_FILE), model, optimizer, LEARNING_RATE)
    for epoch in range(NUM_EPOCHS):
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, DEVICE)
        if SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=CHECKPOINT_FILE)
