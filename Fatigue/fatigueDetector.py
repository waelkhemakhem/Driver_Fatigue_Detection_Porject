# from testor import *
from DetectorUtils import norm

# main functions
# visKeypoints(loadImage(), npData)
# image = loadImage()


def isEyeClosed(eyePoints: list) -> [float, bool]:
    """Detect if the eye is closed

        :param eyePoints: [P1..P6] list of 6 eye key points
        :return: the eye closed value and Boolean, if the eye is closed or not
    """
    EYE_AR_THRESHOLD = 0.3  # if result < threshold then the eye is closed
    p1, p2, p3, p4, p5, p6 = eyePoints
    result: float = (norm(p2, p6) + norm(p3, p5)) / (2 * norm(p1, p4))
    return [result, result < EYE_AR_THRESHOLD]


def isYawning(mouthPoints: list) -> [float, bool]:
    """Detect the yawn state

        :param mouthPoints: [P48..P66] list of 6 mouth keypoints
        :return: the yawn value and Boolean, if the person is yawning
    """
    MOUTH_AR_THRESHOLD = 0.8 # if result < threshold then the eye is closed
    p48, _, _, p51, _, _, p54, _, _, p57, _, _, p60, _, p62, _, p64, _, p66 = mouthPoints
    result: float = ((norm(p51, p57) / norm(p48, p54)) +
                     (norm(p62, p66) / norm(p60, p64))) / 2
    return [result, result > MOUTH_AR_THRESHOLD]


def headDetection():
    """

        :param
        :return:
    """
    pass


# firstEye 37..42 => 38..44
# # secondEye 43..48 => 44..50
# print("=== EYES CLOSED STATE ===")
# val1: bool = isEyeClosed(npData[38:44, :])
# val2: bool = isEyeClosed(npData[44:50, :])
# print(val1, val2)
# print("=== MOUTH OPEN STATE ===")
# # mouth key points 48..66 => 49..68
# val3: bool = isYawning(npData[49:68, :])
# print(val3)
