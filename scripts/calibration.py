import numpy as np
import cv2 as cv
import glob

CHECKERBOARD = (7, 5)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('calibration_images/*.jpg')

if not images:
    print("No images found in the specified path. Please check the path.")
else:
    print(f"Found {len(images)} images for calibration.")

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)

    cv.destroyAllWindows()


    if len(objpoints) > 0 and len(imgpoints) > 0:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("\n--- Calibration Results ---")
        print("Camera matrix:\n", mtx)
        print("\nDistortion coefficient:\n", dist)

        np.savez('camera_calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        print("\nCalibration data saved to 'camera_calibration_data.npz'")


        if images:
            img_to_undistort = cv.imread(images[0])
            h, w = img_to_undistort.shape[:2]

            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

            dst = cv.undistort(img_to_undistort, mtx, dist, None, newcameramtx)

            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            cv.imwrite('calibrated_result.png', dst)
            print("\nAn example of an undistorted image has been saved as 'calibrated_result.png'")

            cv.imshow('Original Image', img_to_undistort)
            cv.imshow('Undistorted Image', dst)
            cv.waitKey(0)
            cv.destroyAllWindows()

    else:
        print("Could not find chessboard corners in enough images for calibration.")