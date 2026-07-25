import cv2
import random
import os

# Tracker作成
def create_tracker(tracker_type):
    if tracker_type == "KCF":
        return cv2.legacy.TrackerKCF_create()
    elif tracker_type == "MIL":
        return cv2.legacy.TrackerMIL_create()
    else:
        return cv2.legacy.TrackerCSRT_create()

def main():
    # Tracker選択
    print("使用するTrackerを選択してください")
    print("1 : CSRT")
    print("2 : KCF")
    print("3 : MIL")

    choice = input("番号を入力 : ")
    if choice == "2":
        tracker_type = "KCF"
    elif choice == "3":
        tracker_type = "MIL"
    else:
        tracker_type = "CSRT"

    # 動画読み込み
    cap = cv2.VideoCapture("moving_peoples.mp4")

    if not cap.isOpened():
        return

    # カスケード分類器読み込み
    cascade_path = os.path.join(
        os.path.dirname(__file__),
        "haarcascade_fullbody.xml"
    )
    cascade = cv2.CascadeClassifier(cascade_path)

    # 最初のフレーム取得
    ret, frame = cap.read()

    if not ret:
        return

    # グレースケール変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 人物検出
    bodies = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(20, 50)
    )

    print("追跡したい人物を複数選択してください")
    print("ENTER または SPACE で決定")
    print("ESCで終了")

    # 複数選択
    bboxes = []
    for (x, y, w, h) in bodies:
        bboxes.append((x, y, w, h))

    # MultiTracker生成
    multi_tracker = cv2.legacy.MultiTracker_create()

    # 色保存
    colors = []

    # Tracker追加
    for bbox in bboxes:
        tracker = create_tracker(tracker_type)
        multi_tracker.add(
            tracker,
            frame,
            bbox
        )
        # ランダム色
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        colors.append(color)

    # 軌跡保存
    trajectories = [[] for _ in range(len(bboxes))]

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        # 追跡更新
        success, boxes = multi_tracker.update(frame)

        # 各人物描画
        for i, newbox in enumerate(boxes):
            x, y, w, h = [int(v) for v in newbox]

            # tracking失敗時
            if w <= 0 or h <= 0:
                continue
            color = colors[i]

            # 矩形描画
            cv2.rectangle(frame,(x, y),(x + w, y + h),color,2)

            # 中心座標
            cx = x + w // 2
            cy = y + h // 2

            # 軌跡保存
            trajectories[i].append((cx, cy))

            # 軌跡描画
            for j in range(1, len(trajectories[i])):
                cv2.line(
                    frame,
                    trajectories[i][j - 1],
                    trajectories[i][j],
                    color,
                    2
                )

        cv2.imshow("multi_tacker_cascade", frame)

        key = cv2.waitKey(30)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()