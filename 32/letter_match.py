import cv2
import numpy as np

# ==========================
# 設定
# ==========================

LETTER_SIZE = 35  # 漢字1文字のサイズ（調整する）
template = None
gray = None
img = None
method = None


# ==========================
# SAD
# ==========================
def match_sad(image, template):

    h, w = template.shape

    result = np.zeros(
        (image.shape[0] - h + 1,
         image.shape[1] - w + 1),
        dtype=np.float32
    )

    for y in range(result.shape[0]):
        for x in range(result.shape[1]):

            roi = image[y:y+h, x:x+w]

            result[y, x] = np.sum(
                np.abs(
                    roi.astype(np.float32)
                    - template.astype(np.float32)
                )
            )

    return result


# ==========================
# テンプレートマッチング
# ==========================
def detect():

    global gray, template, method, img

    if template is None:
        return

    output = img.copy()

    # ------------------
    # SAD
    # ------------------
    if method == "SAD":

        result = match_sad(gray, template)

        threshold = result.min() * 1.3

        locations = np.where(result <= threshold)

    # ------------------
    # SSD
    # ------------------
    elif method == cv2.TM_SQDIFF:

        result = cv2.matchTemplate(
            gray,
            template,
            method
        )

        threshold = result.min() * 1.3

        locations = np.where(result <= threshold)

    # ------------------
    # NCC / ZNCC
    # ------------------
    else:

        result = cv2.matchTemplate(
            gray,
            template,
            method
        )

        locations = np.where(result >= 0.8)

    found = []

    for pt in zip(*locations[::-1]):
        found.append(pt)

    # ------------------
    # 違う漢字を探す
    # ------------------
    for y in range(
        0,
        gray.shape[0] - LETTER_SIZE,
        LETTER_SIZE
    ):

        for x in range(
            0,
            gray.shape[1] - LETTER_SIZE,
            LETTER_SIZE
        ):

            matched = False

            for fx, fy in found:

                if abs(x - fx) < 10 and abs(y - fy) < 10:
                    matched = True
                    break

            if not matched:

                cv2.rectangle(
                    output,
                    (x, y),
                    (x + LETTER_SIZE,
                     y + LETTER_SIZE),
                    (0, 0, 255),
                    3
                )

    cv2.imshow("Result", output)


# ==========================
# マウスイベント
# ==========================
def mouse_event(event, x, y, flags, param):

    global template

    if event == cv2.EVENT_LBUTTONDOWN:

        x1 = max(0, x - LETTER_SIZE // 2)
        y1 = max(0, y - LETTER_SIZE // 2)

        template = gray[
            y1:y1 + LETTER_SIZE,
            x1:x1 + LETTER_SIZE
        ]

        detect()


# ==========================
# メイン
# ==========================
def main():

    global gray, img, method

    print("アルゴリズム選択")
    print("1 : SSD")
    print("2 : SAD")
    print("3 : NCC")
    print("4 : ZNCC")

    mode = int(input("番号 : "))

    if mode == 1:
        method = cv2.TM_SQDIFF

    elif mode == 2:
        method = "SAD"

    elif mode == 3:
        method = cv2.TM_CCORR_NORMED

    else:
        method = cv2.TM_CCOEFF_NORMED

    img = cv2.imread("letter.jpg")

    if img is None:
        print("letter.jpg がありません")
        return

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    cv2.namedWindow("Letter")
    cv2.setMouseCallback(
        "Letter",
        mouse_event
    )

    while True:

        cv2.imshow(
            "Letter",
            img
        )

        key = cv2.waitKey(1)

        if key == 27:
            break

    cv2.destroyAllWindows()


main()