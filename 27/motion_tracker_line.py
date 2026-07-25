import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture("moving_vehicles.mp4")
    print("使用するTrackerを選択")
    print("1 : KCF")
    print("2 : CSRT")
    print("3 : MIL")

    mode = input("番号入力 : ")

    # Tracker選択
    if mode == "1":
        tracker = cv2.TrackerKCF_create()
    elif mode == "2":
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = cv2.TrackerMIL_create()
    tracking = False

    # 軌跡保存用
    points = []

    cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracker", 800, 500)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # tracking中
        if tracking:
            success, roi = tracker.update(frame)

            if success:
                x, y, w, h = map(int, roi)
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 255),
                    2
                )

                # 中心座標
                cx = x + w // 2
                cy = y + h // 2

                # 座標保存
                points.append((cx, cy))

                # 追跡線描画
                for i in range(1, len(points)):
                    cv2.line(
                        frame,
                        points[i - 1],
                        points[i],
                        (0, 255, 255),
                        2
                    )
        cv2.imshow("Tracker", frame)

        # ×ボタンで終了
        if cv2.getWindowProperty("Tracker", cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(30) & 0xFF
        # qで終了
        if key == ord('q'):
            break

        # sで停止 + ROI選択
        elif key == ord('s'):
            # 一時停止画像でROI選択
            roi = cv2.selectROI("Tracker", frame, False)

            # tracker初期化
            tracker.init(frame, roi)

            # tracking開始
            tracking = True

            # 軌跡リセット
            points = []

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()