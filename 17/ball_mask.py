import cv2
import numpy as np

def main():
    color_name = input("色を入力（pink / green / blue）: ")

    img = cv2.imread("./ball.jpg")
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    if color_name == "pink":
        lower = np.array([140, 50, 50])
        upper = np.array([175, 255, 255])
    elif color_name == "green":
        lower = np.array([80, 50, 50])
        upper = np.array([100, 255, 255])
    elif color_name == "blue":
        lower = np.array([100, 50, 50])
        upper = np.array([105, 255, 255])
    else:
        print("その色は未対応です")
        return

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    while True:
        cv2.imshow("Original", img)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1)

        # Escキーで終了
        if key == 27:
            break

        if cv2.getWindowProperty("Original", cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow("Original")

        if cv2.getWindowProperty("Mask", cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow("Mask")

        if (cv2.getWindowProperty("Original", cv2.WND_PROP_VISIBLE) < 1 and
            cv2.getWindowProperty("Mask", cv2.WND_PROP_VISIBLE) < 1):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()