import cv2
import numpy as np

# 色範囲（君の設定そのまま）
color_ranges = {
    "pink": (np.array([140, 50, 50]), np.array([175, 255, 255])),
    "green": (np.array([80, 50, 50]), np.array([100, 255, 255])),
    "blue": (np.array([100, 50, 50]), np.array([105, 255, 255]))
}

clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

# 画像読み込み
img = cv2.imread("ball.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_callback)

while True:
    display = img.copy()

    if clicked_point is not None:
        mask_total = np.zeros(img.shape[:2], dtype=np.uint8)

        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)

            # 輪郭取得
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                # クリック点がこの輪郭の中か？
                result = cv2.pointPolygonTest(cnt, clicked_point, False)

                if result >= 0:  # 内側 or 境界
                    cv2.drawContours(mask_total, [cnt], -1, 255, -1)

        # グレー画像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 合成
        color_part = cv2.bitwise_and(img, img, mask=mask_total)
        gray_part = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(mask_total))

        display = cv2.add(color_part, gray_part)

    cv2.imshow("image", display)

    key = cv2.waitKey(1)
    if key == 27:  # ESCで終了
        break

    if cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()