import torch
from Dataset import Facial_Key_Points
from torch import nn, optim
import os
import Config
from torch.utils.data import DataLoader
from tqdm import tqdm
from Utils import get_images_labels, split_train_val_test
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
        BATCH_SIZE,
        PIN_MEMORY,
        NUM_WORKERS,
        images_path,
        all_data_path,
        train_coef,
        test_coef,
        val_coef
):
    images, labels = get_images_labels(all_data_path)
    X_train, X_test, X_val, y_train, y_test, y_val = split_train_val_test(images, labels, train=train_coef,
                                                                          test=test_coef,
                                                                          val=val_coef)
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


def test(BATCH_SIZE,
         PIN_MEMORY,
         NUM_WORKERS,
         images_path,
         all_data_path,
         train_coef,
         test_coef,
         val_coef,
         CHECKPOINT_FILE,
         model,
         optimizer,
         LEARNING_RATE,
         DEVICE):
    test_loader = Create_train_test_val_loader(
        BATCH_SIZE,
        PIN_MEMORY,
        NUM_WORKERS,
        images_path=images_path,
        all_data_path=all_data_path,
        train_coef=train_coef,
        test_coef=test_coef,
        val_coef=val_coef
    )[2]
    load_checkpoint(torch.load(CHECKPOINT_FILE), model, optimizer, LEARNING_RATE)
    loss_fn = nn.MSELoss(reduction="mean")
    losses = []
    loop = tqdm(test_loader)
    num_examples = 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loop):
            data = X.to(device=DEVICE, dtype=torch.float32)
            targets = y.to(device=DEVICE, dtype=torch.float32)
            scores = model(data)
            loss = loss_fn(scores, targets)
            num_examples += torch.numel(scores[targets != -1])
            losses.append(loss.item())
    print(f"Loss average : {(sum(losses) / num_examples) ** 0.5}")


def train(model,
          DEVICE,
          LEARNING_RATE,
          WEIGHT_DECAY,
          NUM_EPOCHS,
          BATCH_SIZE,
          PIN_MEMORY,
          NUM_WORKERS,
          images_path,
          all_data_path,
          CHECKPOINT_FILE,
          train_coef,
          test_coef,
          val_coef,
          SAVE_MODEL,
          LOAD_MODEL,
          ):
    train_loader, valid_loader, _ = Create_train_test_val_loader(
        BATCH_SIZE,
        PIN_MEMORY,
        NUM_WORKERS,
        images_path=images_path,
        all_data_path=all_data_path,
        train_coef=train_coef,
        test_coef=test_coef,
        val_coef=val_coef
    )
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()
    if LOAD_MODEL and CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(CHECKPOINT_FILE), model, optimizer, LEARNING_RATE)
    for epoch in range(NUM_EPOCHS):
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, DEVICE)
        print("validation of the model")
        # test(model=model, test_loader=valid_loader, loss_fn=loss_fn, device=DEVICE)
        if SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=CHECKPOINT_FILE)
