import cv2
import numpy as np

def main():
    color_name = input("色を入力（pink / green / blue）: ")

    img = cv2.imread("./ball.jpg")
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # 色範囲
    if color_name == "pink":
        lower = np.array([140, 50, 50])
        upper = np.array([175, 255, 255])
    elif color_name == "green":
        lower = np.array([80, 50, 50])
        upper = np.array([100, 255, 255])
    elif color_name == "blue":
        lower = np.array([100, 50, 50])
        upper = np.array([105, 255, 255])
    else:
        print("その色は未対応です")
        return

    # マスク作成（内部処理として使用）
    mask = cv2.inRange(hsv, lower, upper)

    # 輪郭検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # 小さいノイズ除去
        if area < 500:
            continue

        # 外接矩形
        x, y, w, h = cv2.boundingRect(cnt)

        # 黄色で描画
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # ===== 表示 =====
    while True:
        cv2.imshow("Result", img)

        key = cv2.waitKey(1)
        if key == 27:
            break

        if cv2.getWindowProperty("Result", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()