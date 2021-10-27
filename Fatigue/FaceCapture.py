import cv2


def captureFace():
    # load model

    #
    capture = cv2.VideoCapture(0)
    while True:
        has_frame, frame = capture.read()
        if not has_frame:
            print('Can\'t get frame')
            break
        cv2.circle(frame, (100, 100), 2, (0, 255, 0), -1)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(3)
        if key == 27:
            print('Pressed Esc')
            break
    capture.release()
    cv2.destroyAllWindows()


captureFace()
