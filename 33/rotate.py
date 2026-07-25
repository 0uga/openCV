import cv2
import numpy as np

# 画像読み込み
img = cv2.imread("move.png")

h, w = img.shape[:2]

# 元画像の対角線長を計算
size = int(np.sqrt(w**2 + h**2))

# 正方形キャンバス作成
canvas = np.zeros((size, size, 3), dtype=np.uint8)

# 元画像を中央に配置
x = (size - w) // 2
y = (size - h) // 2
canvas[y:y+h, x:x+w] = img

# 回転中心（ウィンド中心）
center = (size // 2, size // 2)

while True:
    for angle in range(360):
        
        # 回転行列
        M = cv2.getRotationMatrix2D(center, angle * -1, 1.0)

        # 回転
        rotated = cv2.warpAffine(
            canvas,
            M,
            (size, size),
            flags=cv2.INTER_LINEAR
        )

        cv2.imshow("Rotate", rotated)

        key = cv2.waitKey(20)

        if key == 27:  # ESCキー
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()