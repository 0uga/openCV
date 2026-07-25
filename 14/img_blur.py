import cv2

def main():
    # 画像読み込み
    img = cv2.imread("./ball.jpg")

    # カーネルサイズ
    ksize1 = 5  # 例：5x5
    ksize2 = 20  # 例：20x20

    # 移動平均フィルタ
    blur1 = cv2.blur(img, (ksize1, ksize1))
    blur2 = cv2.blur(img, (ksize2, ksize2))

    # 表示
    cv2.imshow("Original", img)
    cv2.imshow("Blur1", blur1)
    cv2.imshow("Blur2", blur2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()