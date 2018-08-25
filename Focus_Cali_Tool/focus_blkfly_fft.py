import numpy as np
import cv2
from snark.imaging import cv_image
from copy import deepcopy
import sys

def main():
    for i in cv_image.iterator(sys.stdin):

        # Get image and timestamp
        timestamp = i.header[0]
        timestamp = str(timestamp)
        image = i.data[...]

        # Create a copy of the original for display purposes
        image_copy = deepcopy(image)
        # Convert image to grayscale for processing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Determine 2D fft of image
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        fft_real = np.abs(fshift.real)
        fft_imag = np.abs(fshift.imag)
        num_px = np.prod(image.shape)

        # Get the sum of real spectrum of fft as focus metric
        focus_metric_real = np.sum(fft_real)
        focus_metric_real = (np.sum(focus_metric_real))/num_px

        # Print out the image to be sent through to std for
        # live plotting
        print ("%.2e %s" %(focus_metric_real, timestamp))
        sys.stdout.flush()

        cv2.imshow("Frame", image_copy)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
