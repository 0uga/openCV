import cv2

# 画像の読み込み
img = cv2.imread("poster_2.png")

# AKAZEオブジェクトの作成
akaze = cv2.AKAZE_create()

# 特徴点の抽出
kps, dess = akaze.detectAndCompute(img, None)

# 特徴点を描画
img_akaze = cv2.drawKeypoints(
    img,
    kps,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 表示
cv2.imshow("feature_points", img_akaze)

cv2.waitKey(0)
cv2.destroyAllWindows()