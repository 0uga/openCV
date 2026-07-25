import cv2
import numpy as np

# 画像読み込み
img = cv2.imread("book4.jpg")

# コピー
result = img.copy()

# グレースケール
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二値化
_, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

# 輪郭取得
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 面積最大の輪郭
contour = max(contours,key=cv2.contourArea)

# 四角形になるまで近似
approx = None

for p in np.arange(0.04, 0.2, 0.01):
    epsilon = p * cv2.arcLength(contour, True)
    temp = cv2.approxPolyDP(contour,epsilon,True)

    if len(temp) == 4:
        approx = temp
        break

# 四隅が見つからなかった場合
if approx is None:
    print("四隅が見つかりません")
    exit()

# 座標を取り出す
pts = approx.reshape(4,2).astype(np.float32)

#----------------
# 四隅を並べ替える

s = pts.sum(axis=1)
diff = np.diff(pts, axis=1)

pt1 = pts[np.argmin(s)]      # 左上
pt3 = pts[np.argmax(s)]      # 右下
pt2 = pts[np.argmin(diff)]   # 右上
pt4 = pts[np.argmax(diff)]   # 左下

src = np.array([pt1, pt2, pt3, pt4], dtype=np.float32)


# 出力画像サイズ
width = 600
height = 700

dst = np.array([
    [0,0],
    [width-1,0],
    [width-1,height-1],
    [0,height-1]
], dtype=np.float32)

# 射影変換
M = cv2.getPerspectiveTransform(src, dst)

# 補正
warp = cv2.warpPerspective(img, M, (width,height))

# 頂点表示
for p in src:
    cv2.circle(result,tuple(p.astype(int)),10,(0,0,255),-1)

small = cv2.resize(result, (600, 700))
cv2.imshow("Original", small)
cv2.imshow("Warp", warp)

cv2.waitKey(0)
cv2.destroyAllWindows()