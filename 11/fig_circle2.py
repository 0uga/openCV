import numpy as np
import cv2

def main():
    # 画像読み込み
    img = cv2.imread("./ktech.jpg")

    # 画像サイズ取得
    height, width = img.shape[:2]

    # ===== グリッド線（80ピクセル間隔）=====
    for x in range(0, width, 80):
        cv2.line(img, (x, 0), (x, height), (0, 0, 0), 1)

    for y in range(0, height, 80):
        cv2.line(img, (0, y), (width, y), (0, 0, 0), 1)

    # ===== 平均色 =====
    mean_color = cv2.mean(img)[:3]
    mean_color = tuple(map(int, mean_color))

    # ===== 円 =====
    center = (width // 2, height // 2)
    radius = 100

    # 塗りつぶし（平均色）
    cv2.circle(img, center, radius, mean_color, -1)

    # 円周（白）
    cv2.circle(img, center, radius, (255, 255, 255), 2)

    # 表示
    cv2.imshow("Circle2", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()