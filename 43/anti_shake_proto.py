import cv2
import numpy as np


def main():

    # =============================
    # 動画読み込み
    # =============================
    cap = cv2.VideoCapture("IMG_9040.MOV")

    if not cap.isOpened():
        print("動画を開けませんでした")
        return

    # 動画情報
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("FPS:", fps)
    print("サイズ:", width, "x", height)

    # =============================
    # 出力動画
    # =============================
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        "anti_shake.mp4",
        fourcc,
        fps,
        (width, height)
    )

    # =============================
    # 最初のフレーム
    # =============================
    ret, prev_frame = cap.read()

    if not ret:
        print("最初のフレームを読み込めませんでした")
        cap.release()
        out.release()
        return

    # 最初のフレームを書き込む
    out.write(prev_frame)

    # グレースケール
    prev_gray = cv2.cvtColor(
        prev_frame,
        cv2.COLOR_BGR2GRAY
    )

    # =============================
    # ORB
    # =============================
    orb = cv2.ORB_create(
        nfeatures=1000
    )

    # =============================
    # BFMatcher
    # =============================
    bf = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    frame_count = 1

    # =============================
    # 動画処理
    # =============================
    while True:

        # 次のフレームを取得
        ret, curr_frame = cap.read()

        if not ret:
            print("動画の最後まで到達しました")
            break

        # グレースケール
        curr_gray = cv2.cvtColor(
            curr_frame,
            cv2.COLOR_BGR2GRAY
        )

        # =============================
        # 特徴点検出
        # =============================
        kp1, des1 = orb.detectAndCompute(
            prev_gray,
            None
        )

        kp2, des2 = orb.detectAndCompute(
            curr_gray,
            None
        )

        # 特徴点が取得できなかった場合
        if des1 is None or des2 is None:

            out.write(curr_frame)

            prev_gray = curr_gray
            frame_count += 1

            continue

        # =============================
        # 特徴点マッチング
        # =============================
        matches = bf.match(
            des1,
            des2
        )

        # 距離が小さい順に並べる
        matches = sorted(
            matches,
            key=lambda x: x.distance
        )

        # 上位100個を使用
        matches = matches[:100]

        # マッチ数が少ない場合
        if len(matches) < 4:

            out.write(curr_frame)

            prev_gray = curr_gray
            frame_count += 1

            continue

        # =============================
        # 対応点取得
        # =============================
        src_pts = np.float32([
            kp1[m.queryIdx].pt
            for m in matches
        ])

        dst_pts = np.float32([
            kp2[m.trainIdx].pt
            for m in matches
        ])

        # =============================
        # アフィン変換
        # =============================
        matrix, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC
        )

        # 変換行列が取得できなかった場合
        if matrix is None:

            out.write(curr_frame)

            prev_gray = curr_gray
            frame_count += 1

            continue

        # =============================
        # シフト量
        # =============================
        dx = matrix[0, 2]
        dy = matrix[1, 2]

        # =============================
        # 回転角
        # =============================
        angle = np.arctan2(
            matrix[1, 0],
            matrix[0, 0]
        )

        angle_deg = np.degrees(angle)

        print(
            "フレーム:",
            frame_count,
            "dx:",
            round(dx, 2),
            "dy:",
            round(dy, 2),
            "回転:",
            round(angle_deg, 2)
        )

        # =============================
        # 手ぶれ補正用の変換行列
        # =============================
        correction_matrix = np.array([
            [
                np.cos(angle),
                -np.sin(angle),
                -dx
            ],
            [
                np.sin(angle),
                np.cos(angle),
                -dy
            ]
        ], dtype=np.float32)

        # =============================
        # 手ぶれ補正
        # =============================
        result = cv2.warpAffine(
            curr_frame,
            correction_matrix,
            (width, height)
        )

        # =============================
        # 黒い部分を隠すため少し拡大
        # =============================
        result = cv2.resize(
            result,
            None,
            fx=1.05,
            fy=1.05
        )

        h, w = result.shape[:2]

        x = (w - width) // 2
        y = (h - height) // 2

        result = result[
            y:y + height,
            x:x + width
        ]

        # =============================
        # 動画に書き込み
        # =============================
        out.write(result)

        # =============================
        # 表示
        # =============================
        display = cv2.resize(
            result,
            (800, 600)
        )

        cv2.imshow(
            "Anti Shake",
            display
        )

        # =============================
        # キー入力
        # =============================
        key = cv2.waitKey(1) & 0xFF

        # qキーで終了
        if key == ord("q"):
            print("qキーが押されたため終了します")
            break

        # ×ボタンで終了
        if cv2.getWindowProperty(
            "Anti Shake",
            cv2.WND_PROP_VISIBLE
        ) < 1:
            print("ウィンドウが閉じられたため終了します")
            break

        # =============================
        # 次のフレームへ
        # =============================
        prev_gray = curr_gray
        frame_count += 1

    # =============================
    # 終了処理
    # =============================
    cap.release()
    out.release()

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    print("手ぶれ補正完了")
    print("anti_shake.mp4 に保存しました")


if __name__ == "__main__":
    main()