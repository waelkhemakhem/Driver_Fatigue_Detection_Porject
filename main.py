from train import train, test
from torchvision.models import vgg16
from Utils import plot_image_with_landmarks
from torch import nn, optim
import torch

if __name__ == '__main__':
    train()
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
    #      model=model,
    #      optimizer=optimizer,
    #      LEARNING_RATE=LEARNING_RATE,
    #      DEVICE=DEVICE
    # # #      )
    # plot_image_with_landmarks(checkpoint="KeyPointsModel.pth.tar", model=model, optimizer=optimizer, device=DEVICE,
    #                           lr=LEARNING_RATE, image_path="C:/Users/Utilisateur/Desktop/images/04956.png")
    import cv2

    # image = cv2.imread("C:/Users/Utilisateur/Desktop/images/04999.png")
    # print(image)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
