import cv2

# 円の情報を保存（スタック）
circles = []

def mouse_event(event, x, y, flags, param):
    global circles

    # 左クリック → 円を追加
    if event == cv2.EVENT_LBUTTONDOWN:
        circles.append((x, y))

    # 右クリック → 直前の円を削除
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(circles) > 0:
            circles.pop()  # 最後の円を削除


def main():
    global circles

    # 元画像
    original = cv2.imread("./ktech.jpg")

    cv2.namedWindow("Mouse Circle2")
    cv2.setMouseCallback("Mouse Circle2", mouse_event)

    while True:
        # 毎回コピー
        img = original.copy()

        # 保存されている円をすべて描画
        for (x, y) in circles:
            cv2.circle(img, (x, y), 50, (0, 255, 255), 2)  # 黄色

        cv2.imshow("Mouse Circle2", img)

        key = cv2.waitKey(1)
        if key == 27:
            break

        # ウィンドウ閉じたら終了
        if cv2.getWindowProperty("Mouse Circle2", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()