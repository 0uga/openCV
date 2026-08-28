import cv2

# グローバル変数
drag = False
total_frames = 0
cap = None
playing = True


# マウスイベント
def mouse(event, x, y, flags, param):
    global drag, cap, total_frames, playing

    if event == cv2.EVENT_LBUTTONDOWN:
        if y >= 470:
            drag = True

    elif event == cv2.EVENT_MOUSEMOVE and drag:
        pass

    elif event == cv2.EVENT_LBUTTONUP:
        if drag:
            drag = False

            x = max(0, min(x, 800))
            frame = int(x / 800 * total_frames)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)


def main():

    global cap, total_frames

    cap = cv2.VideoCapture("people_move.mp4")

    if not cap.isOpened():
        print("動画を開けません")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow("original")
    cv2.resizeWindow("original", 800, 500)

    cv2.setMouseCallback("original", mouse)

    playing = True

    ret, frame = cap.read()

    while ret:

        if cv2.getWindowProperty("original", cv2.WND_PROP_VISIBLE) < 1:
            break

        display = cv2.resize(frame, (800, 500))

        # プログレスバー
        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        cv2.rectangle(display, (0, 470), (800, 500), (60, 60, 60), -1)

        pos = int(current / total_frames * 800)

        cv2.rectangle(display, (0, 470), (pos, 500), (0, 255, 0), -1)

        cv2.putText(
            display,
            f"{current}/{total_frames}",
            (10, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("original", display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('p'):
            playing = not playing

        elif key == ord('f'):

            current = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_MSEC, current + 5000)

            ret, frame = cap.read()
            if not ret:
                continue

        # 5秒巻き戻し
        elif key == ord('b'):

            current = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_MSEC, max(current - 5000, 0))

            ret, frame = cap.read()
            if not ret:
                continue

        if playing:
            ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()