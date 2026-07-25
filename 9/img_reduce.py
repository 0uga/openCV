import numpy as np
import cv2

def main():
    img = cv2.imread("./ktech.jpg")

    img_reduce = cv2.resize(img, None, fx=0.5, fy=0.5)

    cv2.imshow("Original", img)
    cv2.imshow("Reduced Image", img_reduce)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()