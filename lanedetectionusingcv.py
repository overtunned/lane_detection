import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def region_int(img):
    height = img.shape[0]
    area = np.array([[(200, height), (1100, height), (550, 220)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, area, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def disp_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_img


def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters
    y1 = img.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def averaged_slope_intercept(img, lines):
    leftfit = []
    rightfit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                leftfit.append((slope, intercept))
            else:
                rightfit.append((slope, intercept))
        left_average = np.average(leftfit, axis=0)
        right_average = np.average(rightfit, axis=0)

        if right_average is not None:
            right_line = make_coordinates(img, right_average)
        if left_average is not None:
            print(left_average)
            left_line = make_coordinates(img, left_average)
    return np.array([left_line, right_line])


def lane_detect(img):
    canny_image = canny(img)
    cropped = region_int(canny_image)
    lines = cv2.HoughLinesP(cropped, 2, np.pi / 180, 100, np.array([]), minLineLength=10, maxLineGap=5)
    averaged = averaged_slope_intercept(img, lines)
    line_image = disp_lines(img, averaged)
    detected = cv2.addWeighted(img, 0.8, line_image, 1, 1)
    return detected


