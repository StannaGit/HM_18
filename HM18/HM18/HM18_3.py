
'''
Приклади реалізації процесів object tracking з використанням методів OpenCV:

https://machinelearningknowledge.ai/learn-object-tracking-in-opencv-python-with-code-examples/
https://docs.opencv.org/3.4/d9/df8/group__tracking.html

1. MeanShift : непараметричний алгоритм, який використовує кольорову гістограму (аналіз кольорового простору) для відстеження об’єктів.
2. CamShift : розширення MeanShift, який адаптується до змін розміру та орієнтації об’єкта.

Package            Version
------------------ -----------
numpy              1.24.1
opencv-python      3.4.18.65

'''

import cv2
import numpy as np

# 1. MeanShift
def MeanShift (cap, scale_percent=50):

    # Read the first frame
    ret, frame = cap.read()

    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame = cv2.resize(frame, (width, height))
    '''
    WARNING! click ENTER
    '''
    # Set the ROI (Region of Interest)
    x, y, w, h = cv2.selectROI(frame)

    # Initialize the tracker
    roi = frame[y:y + h, x:x + w]
    roi_hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 10)


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate the back projection of the histogram
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 256], 1)
        '''
        Apply the MeanShift algorithm
        '''
        ret, track_window = cv2.meanShift(dst, (x, y, w, h), term_crit)

        # Draw the track window on the frame
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', img2)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return

# 2. CamShift
def CamShift (cap, scale_percent=50):

    # Read the first frame
    ret, frame = cap.read()

    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame = cv2.resize(frame, (width, height))

    '''
        WARNING! click ENTER
    '''
    # Set the ROI (Region of Interest)
    x, y, w, h = cv2.selectROI(frame)

    # Initialize the tracker
    roi = frame[y:y + h, x:x + w]
    roi_hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate the back projection of the histogram
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 256], 1)
        '''
        Apply the CamShift algorithm
        '''
        # Apply the CamShift algorithm
        ret, track_window = cv2.CamShift(dst, (x, y, w, h), term_crit)

        # Draw the track window on the frame
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', img2)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return


# -------------------------- Головні виклики -----------------------------------
if __name__ == '__main__':

    # Read the video
    cap = cv2.VideoCapture('chicken.mp4')

    print('Оберіть метод object tracking :')
    print('1 - MeanShift')
    print('2 - CamShift')

    mode = int(input('mode:'))

    if (mode == 1):
        print('1 - MeanShift')
        MeanShift(cap)

    if (mode == 2):
        print('2. CamShift')
        CamShift (cap)




