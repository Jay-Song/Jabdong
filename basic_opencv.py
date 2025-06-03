import cv2
import numpy as np


def main():
    idx = 0
    cam = cv2.VideoCapture(0)

    # sometimes, you need to use v4l2 control to set the camera properties
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # cam.set(cv2.CAP_PROP_FPS, 30)

    while True:
        # Read a frame from the camera
        ret, frame = cam.read()
        if not ret:
            break

        # Display the resulting frame
        cv2.imshow("Color Streams (Press 's' to save, 'q' to quit)", frame)

        # Press 'q' to exit
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

        if key == ord("s"):
            filename = f"./color_{idx:04d}.jpg"
            cv2.imwrite(str(filename), frame)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
