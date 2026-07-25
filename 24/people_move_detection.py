import cv2
import numpy as np

def main():

    cap = cv2.VideoCapture("people_move.mp4")

    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("original", 800, 500)

    cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("gray", 800, 500)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while True:

        if cv2.getWindowProperty("original", cv2.WND_PROP_VISIBLE) < 1:
            break

        diff = cv2.absdiff(frame1, frame2)

        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        thresh = cv2.GaussianBlur(thresh, (5, 5), 0)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area > 1000:

                x, y, w, h = cv2.boundingRect(cnt)

                cv2.rectangle(
                    frame1,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

        # 元動画
        cv2.imshow("original", frame1)

        # 白黒二値化画像
        cv2.imshow("gray", thresh)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        frame1 = frame2
        ret, frame2 = cap.read()

        if not ret:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()