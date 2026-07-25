import numpy as np
import cv2

def main():

    cap = cv2.VideoCapture("people_move.mp4")

    # 再生状態
    playing = True

    # ウィンドウ作成
    speed = 40

    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)

    while True:
        # ×ボタンで終了
        if cv2.getWindowProperty("camera", cv2.WND_PROP_VISIBLE) < 1:
            break
        
        # 再生中のみフレーム更新
        if playing:
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow("camera", frame)

        key = cv2.waitKey(speed) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            playing = False

        elif key == ord('p'):
            playing = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()