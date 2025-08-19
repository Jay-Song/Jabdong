import cv2
import copy
import numpy as np
from scipy.spatial.transform import RigidTransform, Rotation


def main():
    idx = 0
    cam = cv2.VideoCapture(0)

    # sometimes, you need to use v4l2 control to set the camera properties
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cam.set(cv2.CAP_PROP_FPS, 30)

    check_size = 0.06
    check_dims = [4, 5]
    board_w = check_size * check_dims[1]
    board_h = check_size * check_dims[0]
    camera_to_board = RigidTransform.from_components([0.0, 0.0, 0.3], Rotation.from_euler("x", 20.0, True))

    pts = np.float32(
        [
            [board_w * 0.5, board_h * 0.5, 0],
            [-board_w * 0.5, board_h * 0.5, 0],
            [-board_w * 0.5, -board_h * 0.5, 0],
            [board_w * 0.5, -board_h * 0.5, 0],
        ]
    )

    imgpts, _ = cv2.projectPoints(
        pts,
        camera_to_board.rotation.as_rotvec(),
        camera_to_board.translation,
        np.array([[640.0, 0.0, 640.0], [0.0, 640.0, 360.0], [0.0, 0.0, 0.0]]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    imgpts = np.int32(imgpts).reshape(-1, 1, 2)
    print(imgpts)

    while True:
        # Read a frame from the camera
        ret, frame = cam.read()
        if not ret:
            break

        updated = copy.deepcopy(frame)

        # Draw all contours in green with a thickness of 2
        cv2.drawContours(
            updated,
            [imgpts],
            -1,
            (0, 255, 0),
            2,
        )

        # Get frame dimensions
        h, w = updated.shape[:2]

        # Center point
        cx, cy = w // 2, h // 2

        # Draw vertical line (green)
        cv2.line(updated, (cx, 0), (cx, h), (0, 0, 255), 2)

        # Draw horizontal line (red)
        cv2.line(updated, (0, cy), (w, cy), (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow("original", frame)
        # Display the resulting frame
        cv2.imshow("changed", updated)

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
