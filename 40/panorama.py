import cv2
import numpy as np


def main():

    # 画像読み込み
    img1 = cv2.imread("picture1.png")
    img2 = cv2.imread("picture2.png")

    # グレースケール
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT生成
    sift = cv2.SIFT_create()

    # 特徴点抽出
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # BFMatcher作成
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 良いマッチングだけ抽出
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    match_img = cv2.drawMatches(
        img1,kp1,
        img2,kp2,
        good,None,
        flags=2
    )

    # 対応点を取得
    src_pts = []
    dst_pts = []

    for m in good:
        x1 = kp1[m.queryIdx].pt[0]
        y1 = kp1[m.queryIdx].pt[1]

        x2 = kp2[m.trainIdx].pt[0]
        y2 = kp2[m.trainIdx].pt[1]

        src_pts.append((x1, y1))
        dst_pts.append((x2, y2))

    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)

    M, mask = cv2.findHomography(
        dst_pts,
        src_pts,
        cv2.RANSAC,
        5.0
    )

    # パノラマ画像用のサイズ
    h, w = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 画像2を射影変換
    panorama = cv2.warpPerspective(
        img2,
        M,
        (w + w2, max(h, h2))
    )
    panorama[0:h, 0:w] = img1

    match_img = cv2.resize(match_img, None, fx=0.6, fy=0.6)
    panorama = cv2.resize(panorama, None, fx=0.6, fy=0.6)

    #cv2.imshow("Good Matches", match_img)
    cv2.imshow("Panorama", panorama)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()