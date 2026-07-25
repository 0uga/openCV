import cv2

def mouse_event(event, x, y, flags, param):
    img, hsv = param

    if event == cv2.EVENT_LBUTTONDOWN:
        # BGR取得
        b, g, r = img[y, x]

        # HSV取得
        h, s, v = hsv[y, x]

        # コンソール表示
        print(f"座標: ({x}, {y})")
        print(f"BGR: ({b}, {g}, {r})")
        print(f"HSV: ({h}, {s}, {v})")
        print("----------------------")


def main():
    # 画像読み込み
    img = cv2.imread("./ball.jpg")

    # HSVに変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.namedWindow("Color Pickup")

    # imgとhsv両方渡す
    cv2.setMouseCallback("Color Pickup", mouse_event, (img, hsv))

    while True:
        cv2.imshow("Color Pickup", img)

        key = cv2.waitKey(1)
        if key == 27:
            break

        # ウィンドウ閉じたら終了
        if cv2.getWindowProperty("Color Pickup", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()