import cv2


def main():

    # 動画を読み込む
    cap = cv2.VideoCapture("anti_shake_original.mp4")

    if not cap.isOpened():
        print("動画を開けません")
        return

    # FPSを取得
    fps = cap.get(cv2.CAP_PROP_FPS)

    # FPSに合わせて待ち時間を設定
    wait = int(1000 / fps)

    while True:

        # フレームを取得
        ret, frame = cap.read()

        if not ret:
            break

        # 画面に表示
        frame = cv2.resize(frame, (800, 600))

        cv2.imshow(
            "Anti Shake",
            frame
        )

        # qキーまたは×で終了
        key = cv2.waitKey(wait) & 0xff

        if key == ord("q"):
            break

        if cv2.getWindowProperty(
            "Anti Shake",
            cv2.WND_PROP_VISIBLE
        ) < 1:
            break

    # 終了処理
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()