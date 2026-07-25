import numpy as np
import cv2

def main():
    img = cv2.imread("./ktech.jpg")

    nega = 255 - img

    cv2.imshow("Negative", nega)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()