import numpy as np
import cv2

def main():
    img = cv2.imread("./ktech.jpg")

    y1, y2 = 100, 300
    x1, x2 = 200, 400

    mean_val = img.mean(axis=(0, 1)).astype(np.uint8)

    img_mask = img.copy()

    img_mask[y1:y2, x1:x2] = mean_val

    cv2.imshow("Mask Image", img_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()