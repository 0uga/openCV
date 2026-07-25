import numpy as np
import cv2

def main():
    # 画像サイズ
    height = 960
    width = 1280

    # 白いキャンバス
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # グリッド（80ピクセル間隔）
    for x in range(0, width, 80):
        cv2.line(img, (x, 0), (x, height), (0, 0, 0), 1)

    for y in range(0, height, 80):
        cv2.line(img, (0, y), (width, y), (0, 0, 0), 1)

    # 円の設定
    center = (width // 2, height // 2)
    radius = 200

    # 塗りつぶし（黄色）
    cv2.circle(img, center, radius, (0, 255, 255), -1)

    # 円周（青）
    cv2.circle(img, center, radius, (255, 0, 0), 2)

    # ★小さく表示（ここが追加ポイント）
    small = cv2.resize(img, (width // 2, height // 2))

    cv2.imshow("Circle", small)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()