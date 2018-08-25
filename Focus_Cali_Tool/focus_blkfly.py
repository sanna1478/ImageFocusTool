import numpy as np
import cv2
from snark.imaging import cv_image
from copy import deepcopy
import sys

refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    """Call back function when a window of an image opens
    to allow for click events to crop image"""

    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate the cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False


def smooth(x, window_len=11,window='hanning'):
    """Function designed to smooth the signal
    in order to remove unwanted signal noise"""
    s = np.r_[x[window_len-1:10:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat':
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y



def main():
    global refPt, cropping
    start_crop = False

    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", click_and_crop)

    for i in cv_image.iterator(sys.stdin):
        image = i.data[...]
        org_img = deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if start_crop:
            y_1 = roi[0][1]
            y_2 = roi[1][1]

            x_1 = roi[0][0]
            x_2 = roi[1][0]

            # Determine the mid of the rectangle
            y_mid = int((y_2 + y_1)/2)
            cropped_img = image[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
            cropped_line = image[y_mid, x_1:x_2]
            # Determine the mean of the cropped image
            mean_img = np.mean(cropped_img, axis=0)

            # Determine the dt of the mean signal
            signal_dt = np.diff(mean_img)

            # Smooth signal
            smooth_signal = smooth(signal_dt)

            # Get the range metric
            focus_metric = np.max(smooth_signal) - np.min(smooth_signal)

            print (focus_metric)

            cv2.rectangle(org_img, (x_1, y_1), (x_2, y_2), (0,50000,0),3)
            cv2.line(org_img, (x_1,y_mid), (x_2,y_mid), (0,0,50000),4)


        cv2.imshow("frame", org_img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            start_crop = True

            if len(refPt) == 2:
                roi = deepcopy(refPt)
                refPt = []

    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
