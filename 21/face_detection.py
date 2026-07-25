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

# 検出した顔に四角形を描画
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# ウィンドウサイズ変更可能
cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)

# サイズ指定
cv2.resizeWindow("Face Detection", 550, 400)

# 結果表示
cv2.imshow("Face Detection", img)

# 検出人数表示
print("検出人数:", len(faces))

cv2.waitKey(0)
cv2.destroyAllWindows()