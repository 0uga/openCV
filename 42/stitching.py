import cv2


def main():

    # 動画読み込み
    cap = cv2.VideoCapture("IMG_9041.MOV")

    if not cap.isOpened():
        print("動画を開けません")
        return

    images = []

    frame_count = 0
    interval = 30   # 30フレームごとに取得

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 指定間隔で画像取得
        if frame_count % interval == 0:
            # サイズを縮小（処理軽量化）
            frame = cv2.resize(frame, (640, 480))
            images.append(frame)
            print("取得:", frame_count)
        frame_count += 1
    cap.release()

    print("使用画像数:", len(images))

    # Stitcher作成
    stitcher = cv2.Stitcher_create(
        cv2.Stitcher_PANORAMA
    )

    # パノラマ作成
    status, panorama = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        print("パノラマ作成失敗")
        print("エラーコード:", status)
        return

    # 表示
    cv2.imshow("Panorama",panorama)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()