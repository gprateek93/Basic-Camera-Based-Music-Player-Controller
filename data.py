import os
import sys

import cv2
import numpy as np


def record(
    dir
):
    vidObj = cv2.VideoCapture(0)
    frames = []
    while True:
        _, frame = vidObj.read()
        if frame is None:
            break
        frame = cv2.flip(frame, 1)
        frames.append(frame)
        k = cv2.waitKey(2)
        if k == 'q' or k == 27:
            break
        cv2.imshow('Frames', frame)

    h, w, _ = frames[0].shape
    size = (w, h)
    fps = 30
    out = cv2.VideoWriter(dir + '.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'),
                          fps,
                          size)
    for frame in frames:
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()


def remove_background(
    frame,
    bgModel
):
    fgmask = bgModel.apply(frame)
    kernel = np.ones((21, 21), np.uint8)
    # fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    # fgmask = cv2.erode(fgmask, kernel, iterations=1)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def process(
    file_path
):
    blurValue = 41
    threshold = 150
    vidObj = cv2.VideoCapture(file_path)
    frames = []
    back_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=50,
                                                  detectShadows=False)
    while True:
        _, frame = vidObj.read()
        if frame is None:
            break
        #frame = cv2.flip(frame,1)
        # frame = cv2.bilateralFilter(frame, 5, 50, 100)
        img = remove_background(frame, back_sub)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        # cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        k = cv2.waitKey(2)
        if k == 'q' or k == 27:
            break
        # cv2.imshow('Frame', img)
        cv2.imshow('Frames', thresh)


if __name__ == "__main__":
    args = sys.argv[1:]
    record('rg3')
    # process('lg2.avi')
