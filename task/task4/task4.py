import cv2
import numpy as np


def main():

    # 画像読み込み
    img = cv2.imread("seeds.jpg")

    if img is None:
        print("画像読み込み失敗")
        return


    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # 二値化
    # 種を白(255)、背景を黒(0)にする
    ret, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )


    # 膨張・収縮フィルタ
    kernel = np.ones((3, 3), np.uint8)

    # 膨張
    dilate = cv2.dilate(
        binary,
        kernel,
        iterations=1
    )

    # 収縮
    morph = cv2.erode(
        dilate,
        kernel,
        iterations=1
    )


    # 輪郭検出
    contours, hierarchy = cv2.findContours(
        morph,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # 種の数
    count = 0


    # 輪郭ごとに確認
    for cnt in contours:

        # 面積計算
        area = cv2.contourArea(cnt)

        # 小さいノイズを除外
        if area > 100:

            count += 1

            # 種の位置に四角を描画
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )


    print("種の数:", count)


    # -------------------------
    # 表示サイズ縮小
    # -------------------------

    scale = 0.5

    binary_show = cv2.resize(
        binary,
        None,
        fx=scale,
        fy=scale
    )

    morph_show = cv2.resize(
        morph,
        None,
        fx=scale,
        fy=scale
    )

    result_show = cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale
    )


    # 表示
    cv2.imshow("Binary", binary_show)
    cv2.imshow("Morphology", morph_show)
    cv2.imshow("Result", result_show)


    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()