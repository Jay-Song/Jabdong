import cv2
import numpy as np


def main():
    # get the grid and objp
    grid_size = [3, 9]

    diag_spacing_m = 0.025
    spacing = diag_spacing_m / np.sqrt(2)  # in meters

    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    for i in range(grid_size[1]):  # row index
        for j in range(grid_size[0]):  # col index
            idx = i * grid_size[0] + j
            # X = (2*j + (i%2)) * spacing
            # Y = i * spacing
            objp[idx, 0] = (2 * j + (i % 2)) * spacing
            objp[idx, 1] = i * spacing

    mtx = np.array([[960.0, 0.0, 960.0], [0.0, 540.0, 540.0], [0.0, 0.0, 1.0]])

    dist_coeff = np.array(
        [
            -0.05492798984050751,
            0.06234768033027649,
            6.0056820075260475e-05,
            0.0010459973709657788,
            -0.020977558568120003,
        ]
    )

    # Load the image
    img = cv2.imread("./color_0000.jpg", flags=cv2.IMREAD_COLOR)

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
    # print(keypoints)

    # Draw detected blobs as green circles. This helps cv2.findCirclesGrid() .
    img = cv2.drawKeypoints(
        image=img,
        keypoints=keypoints,
        outImage=np.array([]),
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    # cv2.imshow("img", im_with_keypoints)  # display
    # cv2.waitKey(0)
    # im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
    # ret, corners = cv2.findCirclesGrid(
    #     im_with_keypoints, (4, 11), None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID
    # )  # Find the circle grid
    # print(ret)
    # if ret == True:
    #     corners2 = cv2.cornerSubPix(
    #         im_with_keypoints_gray, corners, (11, 11), (-1, -1), criteria
    #     )  # Refines the corner locations.

    # # Draw and display the corners.
    # im_with_keypoints = cv2.drawChessboardCorners(img, (4, 11), corners, ret)

    # cv2.imshow("img", im_with_keypoints)  # display
    # cv2.waitKey(0)

    # # Find the circle grid
    ret, centers = cv2.findCirclesGrid(gray, grid_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector=blobDetector)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    if ret:
        try:
            centers_refined = cv2.cornerSubPix(gray, centers, (11, 11), (-1, -1), criteria=criteria)
            img = cv2.drawChessboardCorners(img, grid_size, centers_refined, ret)
        except Exception as e:
            print("Error when refining corners:", e)
            return

    ret, rvec, tvec = cv2.solvePnP(objp, centers_refined, mtx, dist_coeff)
    if ret:
        img = cv2.drawFrameAxes(img, mtx, dist_coeff, rvec, tvec, 0.1)

    else:
        print("SolvePnP failed")

    # # # Display the resulting frame
    cv2.imshow("", img)

    # Press 'q' to exit
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
