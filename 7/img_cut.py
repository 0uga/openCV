import numpy as np
import cv2

def main():
    img = cv2.imread("./ktech.jpg")

    img_cut = img[100:300, 200:400]

    cv2.imshow("Cut Image", img_cut)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()