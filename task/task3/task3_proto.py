import numpy as np
import cv2

def main():

    cap = cv2.VideoCapture("people_move.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 再生状態
    playing = True

    # 総フレーム数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 再生速度
    speed = 40

    # ウィンドウ作成
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)

    while True:
        # ×ボタンで終了
        if cv2.getWindowProperty("camera", cv2.WND_PROP_VISIBLE) < 1:
            break

        # 再生中のみフレーム更新
        if playing:
            ret, frame = cap.read()

            if not ret:
                print("読み込み失敗")
                playing = False
                continue

            # 動画サイズの変更
            frame = cv2.resize(frame, (800, 600))

            # 現在のフレーム番号
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # プログレスバー設定
            bar_height = 20
            bar_width = frame.shape[1]

            # 再生位置
            progress = int(bar_width * current_frame / total_frames)

            # 背景（灰色）
            cv2.rectangle(
                frame,
                (0, frame.shape[0] - bar_height),
                (bar_width, frame.shape[0]),
                (100, 100, 100),
                -1
            )

            # 再生済み部分（緑色）
            cv2.rectangle(
                frame,
                (0, frame.shape[0] - bar_height),
                (progress, frame.shape[0]),
                (0, 255, 0),
                -1
            )

            # 表示
            cv2.imshow("camera", frame)

        key = cv2.waitKey(speed) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            playing = False

        elif key == ord('p'):
            playing = True

        elif key == ord('f'):

            # 現在フレーム
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 5秒分進める
            skip = int(fps * 5)

            next_frame = current_frame + skip

            # 最後を超えない
            if next_frame >= total_frames:
                next_frame = total_frames - 1

            # 移動
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)

            print("移動先:", next_frame)

            # 移動先のフレームを取得
            ret, frame = cap.read()

            if ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()