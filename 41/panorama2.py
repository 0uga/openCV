import cv2
import numpy as np


def stitch(img1, img2):

    # グレースケール
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT
    sift = cv2.SIFT_create()

    # 特徴点抽出
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 良いマッチだけ
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # 対応点
    src_pts = []
    dst_pts = []

    for m in good:
        src_pts.append(kp1[m.queryIdx].pt)
        dst_pts.append(kp2[m.trainIdx].pt)

    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)

    # ホモグラフィ
    M, mask = cv2.findHomography(
        dst_pts,
        src_pts,
        cv2.RANSAC,
        5.0
    )

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 横長キャンバス
    panorama = cv2.warpPerspective(
        img2,
        M,
        (w1 + w2, max(h1, h2))
    )

    # img1を配置
    roi = panorama[0:h1, 0:w1]

    mask1 = np.sum(roi, axis=2) > 0
    mask2 = np.sum(img1, axis=2) > 0

    overlap = mask1 & mask2

    # 重なっている部分を平均化
    roi[overlap] = (
        roi[overlap].astype(np.float32) +
        img1[overlap].astype(np.float32)
    ) / 2

    # 重なっていない部分だけコピー
    roi[mask2 & ~mask1] = img1[mask2 & ~mask1]

    panorama[0:h1, 0:w1] = roi

    # 黒い余白を除去
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(contours[0])
        panorama = panorama[y:y+h, x:x+w]

    return panorama


def main():

    img1 = cv2.imread("hirosawa1.jpg")
    img2 = cv2.imread("hirosawa2.jpg")
    img3 = cv2.imread("hirosawa3.jpg")

    # 1枚目＋2枚目
    pano12 = stitch(img1, img2)

    # さらに3枚目
    panorama = stitch(pano12, img3)

    panorama = cv2.resize(
        panorama,
        None,
        fx=0.5,
        fy=0.5
    )

    cv2.imshow("Panorama", panorama)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()