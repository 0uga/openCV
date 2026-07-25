import cv2

# カスケード分類器の読み込み
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# 画像の読み込み
img = cv2.imread("face2.jpg")

# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 顔検出
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(30, 30)
)

# 顔部分にモザイク処理
for (x, y, w, h) in faces:

    # 顔部分を切り出し
    face = img[y:y+h, x:x+w]

    # 縮小
    small = cv2.resize(face, (20, 20))

    # 拡大（モザイク化）
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # 元画像に戻す
    img[y:y+h, x:x+w] = mosaic

# ウィンドウ設定
cv2.namedWindow("Face Mosaic", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Mosaic", 550, 400)

# 表示
cv2.imshow("Face Mosaic", img)

print("検出人数:", len(faces))

cv2.waitKey(0)
cv2.destroyAllWindows()