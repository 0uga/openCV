import cv2
import numpy as np


def main():

    # 動画を読み込む
    cap = cv2.VideoCapture("IMG_9040.MOV")

    # 動画情報を取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("FPS:", fps)
    print("サイズ:", w, "x", h)

    # 出力動画を作成
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "anti_shake.mp4",
        fourcc,
        fps,
        (w, h)
    )

    if not out.isOpened():
        print("出力動画を作成できません")
        cap.release()
        return

    # 最初のフレーム
    ret, prev = cap.read()

    if not ret:
        print("動画を読み込めません")
        cap.release()
        out.release()
        return

    # 最初のフレームも保存
    out.write(prev)

    # ORB
    orb = cv2.ORB_create(nfeatures=1000)

    # BFMatcher
    bf = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    frame_count = 1

    while True:

        # 次のフレーム
        ret, curr = cap.read()

        if not ret:
            break

        # グレースケール
        gray1 = cv2.cvtColor(
            prev,
            cv2.COLOR_BGR2GRAY
        )

        gray2 = cv2.cvtColor(
            curr,
            cv2.COLOR_BGR2GRAY
        )

        # 特徴点を検出
        kp1, des1 = orb.detectAndCompute(
            gray1,
            None
        )

        kp2, des2 = orb.detectAndCompute(
            gray2,
            None
        )

        # 特徴点が取得できなかった場合
        if des1 is None or des2 is None:
            result = curr

        else:
            # 特徴点をマッチング
            matches = bf.match(
                des1,
                des2
            )

            # 距離の小さい順
            matches = sorted(
                matches,
                key=lambda x: x.distance
            )[:100]

            if len(matches) < 4:
                result = curr

            else:
                # 対応点
                src = np.float32([
                    kp1[m.queryIdx].pt
                    for m in matches
                ])

                dst = np.float32([
                    kp2[m.trainIdx].pt
                    for m in matches
                ])

                # アフィン変換
                M, _ = cv2.estimateAffinePartial2D(
                    src,
                    dst,
                    method=cv2.RANSAC
                )

                if M is None:
                    result = curr

                else:
                    # シフト
                    dx = M[0, 2]
                    dy = M[1, 2]

                    # 回転
                    angle = np.arctan2(
                        M[1, 0],
                        M[0, 0]
                    )

                    # 逆方向に補正
                    correction = np.array([
                        [
                            np.cos(angle),
                            -np.sin(angle),
                            -dx
                        ],[
                            np.sin(angle),
                            np.cos(angle),
                            -dy
                        ]
                    ], dtype=np.float32)

                    # 手ぶれ補正
                    result = cv2.warpAffine(
                        curr,
                        correction,
                        (w, h)
                    )

        # 補正したフレームを保存
        out.write(result)
        frame_count += 1

        # 進捗表示
        if frame_count % 50 == 0:
            print("処理中:",frame_count,"/",1100)

        # 次のフレームへ
        prev = curr

    # 終了
    cap.release()
    out.release()

    print("処理したフレーム数:", frame_count)
    print("手ぶれ補正動画を作成しました")


if __name__ == "__main__":
    main()