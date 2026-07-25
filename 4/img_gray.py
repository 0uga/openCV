import numpy as np
import cv2

def main():
    img = cv2.imread("./ktech.jpg")

    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    gray_mean = ((B + G + R) / 3).astype(np.uint8)

    gray_lumi = (0.114 * B + 0.587 * G + 0.299 * R).astype(np.uint8)

    cv2.imshow("Gray Mean", gray_mean)
    cv2.imshow("Gray Luminance", gray_lumi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()