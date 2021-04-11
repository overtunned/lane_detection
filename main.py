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


if __name__ == "__main__":
    image = cv2.imread('test_image.jpg')
    lane_image = np.copy(image)
    canny_image = canny(lane_image)
    cropped = region_int(canny_image)
    lines = cv2.HoughLinesP(cropped, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_image = disp_lines(cropped, lines)
    imposed = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    cv2.imshow('image', line_image)
    cv2.waitKey(0)
