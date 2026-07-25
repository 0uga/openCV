import cv2


def main():

    img = cv2.imread("books.jpg")

    # 平滑化
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # グレー化
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # 二値化
    _, binary = cv2.threshold(
        gray, 115, 255, cv2.THRESH_BINARY_INV
    )

    # 輪郭抽出
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 輪郭描画
    result = img.copy()
    cv2.drawContours(
        result,
        contours,
        -1,
        (0, 255, 255),
        2
    )

    # 表示サイズ縮小
    scale = 0.5
    gray = cv2.resize(gray, None, fx=scale, fy=scale)
    binary = cv2.resize(binary, None, fx=scale, fy=scale)
    result = cv2.resize(result, None, fx=scale, fy=scale)

    # 表示
    cv2.imshow("book_extraction", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()