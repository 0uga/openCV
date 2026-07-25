import cv2
import numpy as np


def main():

    # 画像読み込み
    books = cv2.imread("books.jpg")
    book = cv2.imread("book5.jpg")

    # グレースケール
    gray_books = cv2.cvtColor(books, cv2.COLOR_BGR2GRAY)
    gray_book = cv2.cvtColor(book, cv2.COLOR_BGR2GRAY)

    # ORB特徴点検出
    orb = cv2.ORB_create()

    kps1, des1 = orb.detectAndCompute(gray_book, None)
    kps2, des2 = orb.detectAndCompute(gray_books, None)

    # マッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(des1, des2, k=2)


    # 良いマッチだけ残す
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # books側の対応点
    points = []
    for m in good:
        x, y = kps2[m.trainIdx].pt
        points.append((int(x), int(y)))

    # 輪郭抽出用に二値化
    _, binary = cv2.threshold(
        gray_books, 120, 255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    max_count = 0
    selected = None


    # 各本の輪郭を調べる
    for cnt in contours:
        count = 0
        for p in points:
            # 点が輪郭内部ならカウント
            inside = cv2.pointPolygonTest(cnt,p,False)
            if inside >= 0:
                count += 1
        # 最大数更新
        if count > max_count:
            max_count = count
            selected = cnt


    # 選択された本を四角で囲む
    if selected is not None:
        x, y, w, h = cv2.boundingRect(selected)
        cv2.rectangle(
            books,(x, y),(x+w, y+h),(0,255,255),3)


    # 表示サイズ縮小
    h, w = books.shape[:2]
    scale = 800 / w
    books = cv2.resize(
        books,
        (int(w*scale), int(h*scale))
    )


    cv2.imshow(
        "book_selection",
        books
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()