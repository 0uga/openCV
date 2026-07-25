import cv2
import numpy as np

# 画像読み込み
img = cv2.imread("poster.jpg")

# 補正前の四隅
pt1 = (85, 52)
pt2 = (228, 115)
pt3 = (237, 332)
pt4 = (103, 422)

# 出力画像サイズ
w = 300
h = 300
d = 100

# 補正後の四隅
pp1 = (d, d)
pp2 = (w + d, d)
pp3 = (w + d, h + d)
pp4 = (d, h + d)

# 座標配列
p_original = np.float32([pt1, pt2, pt3, pt4])
p_trans = np.float32([pp1, pp2, pp3, pp4])

# 射影変換行列
M = cv2.getPerspectiveTransform(p_original, p_trans)

# 射影変換
result = cv2.warpPerspective(
    img,
    M,
    (w + 2 * d, h + 2 * d)
)

# 表示
cv2.imshow("Original", img)
cv2.imshow("Poster", result)

cv2.waitKey(0)
cv2.destroyAllWindows()