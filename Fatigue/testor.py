import cv2
import numpy as np
import json


def getDataArray():
    with open('all_data.json') as f:
        data = json.load(f)
        return data


img_url = "00018.png"
img_num = "18"

npData = getDataArray()
npData = np.asarray(npData[img_num]["face_landmarks"])


def loadImage():
    return cv2.imread(img_url)


def visKeypoints(image, keypoints, diameter=2):
    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for point in keypoints:
        x, y = point[0], point[1]
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# calculate fatigue
# npData: landmarks
# loadImage() : image
