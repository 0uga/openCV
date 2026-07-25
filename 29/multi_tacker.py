import cv2
import random

# Tracker作成関数
def create_tracker():
    return cv2.legacy.TrackerCSRT_create()

def main():

    # 動画読み込み
    cap = cv2.VideoCapture("moving_peoples.mp4")

    if not cap.isOpened():
        return

    # 最初のフレーム取得
    ret, frame = cap.read()

    if not ret:
        return

    # MultiTracker生成
    multi_tracker = cv2.legacy.MultiTracker_create()

    # 色リスト
    colors = []

    print("追跡したい人物を複数選択してください")
    print("選択後 ENTER または SPACE")
    print("終了するとき ESC")

    # 複数ROI選択
    bboxes = cv2.selectROIs("multi_tacker", frame, False)


    # Tracker追加
    for bbox in bboxes:
        tracker = create_tracker()
        multi_tracker.add(tracker, frame, bbox)

        # ランダム色
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        colors.append(color)

    # 軌跡保存用
    trajectories = [[] for _ in range(len(bboxes))]

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 追跡更新
        success, boxes = multi_tracker.update(frame)

        # 各人物描画
        if success:
           for i, newbox in enumerate(boxes):
            x, y, w, h = [int(v) for v in newbox]

            color = colors[i]

            # 矩形描画
            cv2.rectangle(frame,
                          (x, y),
                          (x + w, y + h),
                          color,
                          2)

            # 中心座標
            cx = x + w // 2
            cy = y + h // 2

            # 軌跡保存
            trajectories[i].append((cx, cy))

            # 軌跡描画
            for j in range(1, len(trajectories[i])):
                cv2.line(frame,
                         trajectories[i][j - 1],
                         trajectories[i][j],
                         color,
                         2)

        cv2.imshow("multi_tacker", frame)

        key = cv2.waitKey(30)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()