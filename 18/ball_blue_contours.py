import cv2
import numpy as np

def main():
    # 画像読み込み
    img = cv2.imread("./ball.jpg")

    # 平滑化
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # HSV変換
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # 青色の範囲
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # マスク作成
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 輪郭検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭描画（黄色）
    cv2.drawContours(img, contours, -1, (0, 255, 255), 2)

    # 表示
    cv2.imshow("Original", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()