import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture("moving_vehicles.mp4")
    playing = True
    roi = None

    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI", 800, 500)

    while True:
        if playing:
            ret, frame = cap.read()
            if not ret:
                break

        # ROIが設定されている場合
        if roi is not None:
            x, y, w, h = roi

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                2
            )
        cv2.imshow("ROI", frame)

        # ×ボタンで終了
        if cv2.getWindowProperty("ROI", cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(30) & 0xFF
        # qキーで終了
        if key == ord('q'):
            break

        # sキーで停止 + ROI選択
        elif key == ord('s'):

            playing = False

            roi = cv2.selectROI("ROI", frame, False)
            # ROI確定後に自動再開
            playing = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()