import cv2
from lanedetectionusingcv import *

if __name__ == "__main__":
    # image = cv2.imread('test_image.jpg')
    # lane_image = np.copy(image)
    # imposed = lane_detect(lane_image)
    # cv2.imshow('image', imposed)
    # cv2.waitKey(0)

    cap = cv2.VideoCapture('test2.mp4')
    while cap.isOpened():
        _, frame = cap.read()
        canny_image = canny(frame)
        cropped = region_int(canny_image)
        lines = cv2.HoughLinesP(cropped, 2, np.pi / 180, 100, np.array([]), minLineLength=10, maxLineGap=5)
        averaged = averaged_slope_intercept(frame, lines)
        line_image = disp_lines(frame, averaged)
        detected = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow('image', detected)
        if cv2.waitKey(1) == ord('q'):
            break
