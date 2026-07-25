import numpy as np
import cv2

def main():
    img = cv2.imread("./ktech.jpg")

    img_v_flip = cv2.flip(img, 0)

    img_h_flip = cv2.flip(img, 1)

    cv2.imshow("Vertical Flip", img_v_flip)
    cv2.imshow("Horizontal Flip", img_h_flip)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()