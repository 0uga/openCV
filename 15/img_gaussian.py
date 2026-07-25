import cv2

def main():
    # 画像読み込み
    img = cv2.imread("./ball.jpg")

    # カーネルサイズ（奇数にする）
    ksize1 = 5  # 例：5x5
    ksize2 = 15  # 例：20x20

    # ガウシアンフィルタ（sigmaX = 0）
    gaussian1 = cv2.GaussianBlur(img, (ksize1, ksize1), 0)
    gaussian2 = cv2.GaussianBlur(img, (ksize2, ksize2), 0)

    # 表示
    cv2.imshow("Original", img)
    cv2.imshow("Gaussian1", gaussian1)
    cv2.imshow("Gaussian2", gaussian2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()