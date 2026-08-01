import cv2
import numpy as np


def main():

    # 色範囲
    color_ranges = {
        "pink": (np.array([140, 50, 50]), np.array([175, 255, 255])),
        "green": (np.array([80, 50, 50]), np.array([100, 255, 255])),
        "blue": (np.array([100, 50, 50]), np.array([105, 255, 255]))
    }

    # 画像読み込み
    img = cv2.imread("ball.jpg")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 全ボールのマスク
    mask_total = np.zeros(img.shape[:2], dtype=np.uint8)

    for lower, upper in color_ranges.values():

        mask = cv2.inRange(hsv, lower, upper)

        # 輪郭取得
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # ボールを塗りつぶす
        cv2.drawContours(mask_total, contours, -1, 255, -1)

    # グレー画像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 合成
    color_part = cv2.bitwise_and(img, img, mask=mask_total)
    gray_part = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(mask_total))

    result = cv2.add(color_part, gray_part)

    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()