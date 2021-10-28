from main import train, test
from Utils import get_mean_std, plot_image_with_landmarks
from torchvision.models import vgg16
from torch import nn, optim
import torch
from efficientnet_pytorch import EfficientNet
import cv2

BATCH_SIZE = 64
PIN_MEMORY = True
NUM_WORKERS = 0
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-5
###############################################################
images_path = "../images/"
image_to_plot = "../images/00018.png"
all_data_path = "all_data.json"
CHECKPOINT_FILE = "../efficientnet-b0_v2-KeyPointsModel.pth.tar"
pretrained_model_name = "efficientnet-b0"
###############################################################
DEVICE = 'cpu'
model = EfficientNet.from_pretrained(pretrained_model_name)
model._fc = nn.Linear(in_features=1280, out_features=68 * 2, bias=True)
model = model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
train_coef = 0.8
test_coef = 0.1
val_coef = 0.1
SAVE_MODEL = True
LOAD_MODEL = True

if __name__ == "__main__":
    plot_image_with_landmarks(
        checkpoint=CHECKPOINT_FILE,
        model=model, optimizer=optimizer, device=DEVICE,
        lr=LEARNING_RATE,
        image_path=image_to_plot)


def trainTheModel():
    train(model,
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
          )
    # image = cv2.imread("D:/ENSI/3eme/Aprentissage_supervisé/Driver_fatigue_detection_project/images/01082.png")
    # # let's resize our image to be 150 pixels wide, but in order to
    # print(image)
    # plot_image_with_landmarks(
    #     checkpoint=CHECKPOINT_FILE,
    #     model=model, optimizer=optimizer, device=DEVICE,
    #     lr=LEARNING_RATE,
    #
    #     image_path="D:/ENSI/3eme/Aprentissage_supervisé/Driver_fatigue_detection_project/images/01082.png")
    # train(model=model,
    #       DEVICE=DEVICE,
    #       LEARNING_RATE=LEARNING_RATE,
    #       WEIGHT_DECAY=WEIGHT_DECAY,
    #       NUM_EPOCHS=NUM_EPOCHS,
    #       BATCH_SIZE=BATCH_SIZE,
    #       PIN_MEMORY=PIN_MEMORY,
    #       NUM_WORKERS=NUM_WORKERS,
    #       images_path=images_path,
    #       all_data_path=all_data_path,
    #       CHECKPOINT_FILE=CHECKPOINT_FILE,
    #       train_coef=train_coef,
    #       test_coef=test_coef,
    #       val_coef=val_coef,
    #       SAVE_MODEL=SAVE_MODEL,
    #       LOAD_MODEL=LOAD_MODEL,
    #       )

# test(BATCH_SIZE=BATCH_SIZE,
#      PIN_MEMORY=PIN_MEMORY,
#      NUM_WORKERS=NUM_WORKERS,
#      LEARNING_RATE=LEARNING_RATE,
#      images_path=images_path,
#      all_data_path=all_data_path,
#      CHECKPOINT_FILE=CHECKPOINT_FILE,
#      DEVICE=DEVICE,
#      model=model,
#      optimizer=optimizer,
#      train_coef=train_coef,
#      test_coef=test_coef,
#      val_coef=val_coef)

# pretrained model
# model = EfficientNet.from_pretrained("efficientnet-b7")
# model._fc = nn.Linear(in_features=1280, out_features=68 * 2, bias=True)
# model = model.to(DEVICE)
# model = EfficientNet.from_pretrained("efficientnet-b0")
# model._fc = nn.Linear(in_features=1280, out_features=68 * 2, bias=True)
# Data_loader = Create_train_test_val_loader(BATCH_SIZE=64,
#                                            PIN_MEMORY=True,
#                                            NUM_WORKERS=0,
#                                            images_path="D:/ENSI/3eme/Aprentissage_supervisé/Driver_fatigue_detection_project/images/",
#                                            all_data_path="D:/ENSI/3eme/Aprentissage_supervisé/Driver_fatigue_detection_project/all_data.json",
#                                            train=0.1,
#                                            test=0.8,
#                                            val=0.1)
# mean, std = get_mean_std(Data_loader[2])
# print("le mean est :", mean)
# print("le std est :", std)

# train(DEVICE="cuda" if torch.cuda.is_available() else 'cpu',
#       LEARNING_RATE=1e-4,
#       WEIGHT_DECAY=5e-5,
#       NUM_EPOCHS=20,
#       BATCH_SIZE=64,
#       PIN_MEMORY=True,
#       NUM_WORKERS=0,
#       images_path="D:/ENSI/3eme/Aprentissage_supervisé/Driver_fatigue_detection_project/images/",
#       all_data_path="D:/ENSI/3eme/Aprentissage_supervisé/Driver_fatigue_detection_project/all_data.json",
#       CHECKPOINT_FILE="KeyPointsModel.pth.tar",
#       SAVE_MODEL=True,
#       LOAD_MODEL=False,
#       train=0.8,
#       test=0.1,
#       val=0.1
#       )
# LEARNING_RATE = 1e-4
# WEIGHT_DECAY = 5e-5
# DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
# # model = EfficientNet.from_pretrained("efficientnet-b0")
# # model._fc = nn.Linear(in_features=1280, out_features=68 * 2, bias=True)
# # model = model.to(DEVICE)
# model = vgg16(pretrained=False)
# for param in model.parameters():
#     param.requires_grad = False
#
# model.classifier[6] = nn.Linear(4096, 68 * 2)
#
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# # test(images_path="C:/Users/Utilisateur/Desktop/images/",
# #      all_data_path="C:/Users/Utilisateur/Desktop/all_data.json",
# #      CHECKPOINT_FILE="KeyPointsModel.pth.tar",
# # #      model=model,
# # #      optimizer=optimizer,
# # #      LEARNING_RATE=LEARNING_RATE,
# # #      DEVICE=DEVICE
# # # # #      )

# print("============================")


# # import cv2

# image = cv2.imread("C:/Users/Utilisateur/Desktop/images/04999.png")
# print(image)
# cv2.imshow("image", image)
# cv2.waitKey(0)
