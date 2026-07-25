import cv2

# 複数ROI選択
def multiROIs(window_name, frame):

    rois = []

    while True:
        temp = frame.copy()

        # 過去ROI表示
        for (x, y, w, h) in rois:
            cv2.rectangle(
                temp,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                2
            )

        # ROI選択
        roi = cv2.selectROI(window_name, temp, False)
        x, y, w, h = roi

        # Enterのみで終了
        if w == 0 or h == 0:
            break
        rois.append(roi)
    return rois


def main():
    cap = cv2.VideoCapture("moving_vehicles.mp4")
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video", 900, 600)

    paused = False

    # tracker保存用
    trackers = []

    while True:
        if not paused:
            ret, frame = cap.read()

            if not ret:
                break
        display = frame.copy()

        # tracker更新
        for tracker in trackers:

            success, roi = tracker.update(frame)
            if success:
                x, y, w, h = map(int, roi)
                cv2.rectangle(
                    display,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 255),
                    2
                )
        cv2.imshow("Video", display)

        # ×ボタン終了
        if cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(30) & 0xFF

        # q終了
        if key == ord('q'):
            break

        # sで停止＋ROI選択
        elif key == ord('s'):
            paused = True
            rois = multiROIs("Video", frame)

            # tracker初期化
            trackers = []
            for roi in rois:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, roi)
                trackers.append(tracker)

            # 再開
            paused = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()