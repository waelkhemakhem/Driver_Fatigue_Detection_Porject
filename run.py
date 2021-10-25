from main import Create_train_test_val_loader, train
from Utils import get_mean_std, plot_image_with_landmarks
from torchvision.models import vgg16
from torch import nn, optim
import torch
from efficientnet_pytorch import EfficientNet

if __name__ == '__main__':
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

    train(DEVICE="cuda" if torch.cuda.is_available() else 'cpu',
          LEARNING_RATE=1e-4,
          WEIGHT_DECAY=5e-5,
          NUM_EPOCHS=20,
          BATCH_SIZE=64,
          PIN_MEMORY=True,
          NUM_WORKERS=0,
          images_path="D:/ENSI/3eme/Aprentissage_supervisé/Driver_fatigue_detection_project/images/",
          all_data_path="D:/ENSI/3eme/Aprentissage_supervisé/Driver_fatigue_detection_project/all_data.json",
          CHECKPOINT_FILE="KeyPointsModel.pth.tar",
          SAVE_MODEL=True,
          LOAD_MODEL=False,
          train=0.8,
          test=0.1,
          val=0.1
          )
    # LEARNING_RATE = 1e-4
    # WEIGHT_DECAY = 5e-5
    # DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
    # model = vgg16(pretrained=False)
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.classifier[6] = nn.Linear(in_features=4096, out_features=68 * 2, bias=True)
    # print(model)
    # model = model.to(DEVICE)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # test(images_path="C:/Users/Utilisateur/Desktop/images/",
    #      all_data_path="C:/Users/Utilisateur/Desktop/all_data.json",
    #      CHECKPOINT_FILE="KeyPointsModel.pth.tar",
    # #      model=model,
    # #      optimizer=optimizer,
    # #      LEARNING_RATE=LEARNING_RATE,
    # #      DEVICE=DEVICE
    # # # #      )
    # plot_image_with_landmarks(checkpoint="D:/ENSI/3eme/Aprentissage_supervisé/Driver_fatigue_detection_project/models/fisrtVGG16KeyPointsModel.pth.tar", model=model, optimizer=optimizer, device=DEVICE,
    #                           lr=LEARNING_RATE,
    #                           image_path="D:/ENSI/3eme/Aprentissage_supervisé/Driver_fatigue_detection_project/images/04756.png")
    import cv2

    # image = cv2.imread("C:/Users/Utilisateur/Desktop/images/04999.png")
    # print(image)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
