import cv2

# マウスイベント用関数
def mouse_event(event, x, y, flags, param):
    img = param

    # 左クリック → 赤い円
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 50, (0, 0, 255), 2)

    # 右クリック → 青い円
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img, (x, y), 50, (255, 0, 0), 2)


def main():
    # 画像読み込み
    img = cv2.imread("./ktech.jpg")

    # ウィンドウ作成
    cv2.namedWindow("Mouse Circle")

    # マウスイベント登録（imgを渡す）
    cv2.setMouseCallback("Mouse Circle", mouse_event, img)

    while True:
        # 表示
        cv2.imshow("Mouse Circle", img)

        # キー入力待ち（Escで終了）
        key = cv2.waitKey(1)
        if key == 27:
            break

        # ウィンドウ閉じたら終了
        if cv2.getWindowProperty("Mouse Circle", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()