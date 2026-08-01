import cv2
import numpy as np

def main():

    # カスケード分類器
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # 画像読み込み
    img = cv2.imread("faces.jpg")

    # グレースケール
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 顔検出
    faces = cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=10,
minSize=(140, 140)
)

    print("検出人数:", len(faces))

    # 1枚の大きさ
    size = 100

    # 1行4枚
    cols = 4
    rows = (len(faces) + cols - 1) // cols

    # 顔リスト画像
    face_list = np.ones((rows * size, cols * size, 3),
                        dtype=np.uint8) * 255

    # 顔を貼り付ける
    for i, (x, y, w, h) in enumerate(faces):

        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (size, size))

        row = i // cols
        col = i % cols

        face_list[row*size:(row+1)*size,
                  col*size:(col+1)*size] = face

    cv2.imshow("Face List", face_list)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()