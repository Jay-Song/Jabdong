import cv2
import numpy as np


def main():
    img = cv2.imread("./color_0000.jpg", flags=cv2.IMREAD_COLOR)

    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grid_size = [4, 11]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Setup SimpleBlobDetector parameters.
    blobParams = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    blobParams.minThreshold = 8
    blobParams.maxThreshold = 200

    # Filter by Area.
    blobParams.filterByArea = True
    blobParams.minArea = 70  # minArea may be adjusted to suit for your experiment
    blobParams.maxArea = 2500  # maxArea may be adjusted to suit for your experiment

    # Filter by Circularity
    blobParams.filterByCircularity = True
    blobParams.minCircularity = 0.2

    # Filter by Convexity
    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.87

    # Create a detector with the parameters
    blobDetector = cv2.SimpleBlobDetector_create(blobParams)

    keypoints = blobDetector.detect(gray)  # Detect blobs.
    print(len(keypoints))

    # # Draw detected blobs as red circles. This helps cv2.findCirclesGrid() .
    # im_with_keypoints = cv2.drawKeypoints(
    #     img,
    #     keypoints,
    #     np.array([]),
    #     (0, 255, 0),
    #     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    # )
    # cv2.imshow("img", im_with_keypoints)  # display
    # cv2.waitKey(0)
    # im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
    # ret, corners = cv2.findCirclesGrid(
    # im_with_keypoints, (4, 11), None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID
    # )  # Find the circle grid
    # print(ret)
    # if ret == True:
    # corners2 = cv2.cornerSubPix(
    #     im_with_keypoints_gray, corners, (11, 11), (-1, -1), criteria
    # )  # Refines the corner locations.

    # Draw and display the corners.
    #     im_with_keypoints = cv2.drawChessboardCorners(img, (4, 11), corners, ret)

    # cv2.imshow("img", im_with_keypoints)  # display
    # cv2.waitKey(0)

    # # Find the circle grid
    found, centers = cv2.findCirclesGrid(
        gray, grid_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector=blobDetector
    )

    cv2.drawChessboardCorners(img, grid_size, centers, found)
    # # Display the resulting frame
    cv2.imshow("", img)

    # Press 'q' to exit
    key = cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
