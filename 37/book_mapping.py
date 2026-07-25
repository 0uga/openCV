import cv2

# 画像読み込み
img1 = cv2.imread("book5.jpg")     # 特定の本
img2 = cv2.imread("books.jpg")     # 複数の本

# グレースケール
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# AKAZE
akaze = cv2.AKAZE_create()

# 特徴点抽出
kps1, dess1 = akaze.detectAndCompute(gray1, None)
kps2, dess2 = akaze.detectAndCompute(gray2, None)

# BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# マッチング
matches = bf.knnMatch(dess1, dess2, k=2)

# Ratio Test
ratio = 0.8
good_matches = []

for m, n in matches:
    if m.distance < ratio * n.distance:
        good_matches.append([m])

# マッチング結果を描画
img_matches = cv2.drawMatchesKnn(
    img1,
    kps1,
    img2,
    kps2,
    good_matches,
    None,
    flags=2
)

# 画像を50%に縮小
img_small = cv2.resize(img_matches, None, fx=0.5, fy=0.5)

# 表示
cv2.imshow("Book Mapping", img_small)
cv2.waitKey(0)
cv2.destroyAllWindows()