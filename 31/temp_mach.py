import cv2

print("1 : SSD")
print("2 : SAD")
print("3 : NCC")
print("4 : ZNCC")

choice = input("アルゴリズム選択：")

methods = {
    "1": ("SSD", cv2.TM_SQDIFF),
    "2": ("SAD", cv2.TM_SQDIFF_NORMED),
    "3": ("NCC", cv2.TM_CCORR_NORMED),
    "4": ("ZNCC", cv2.TM_CCOEFF_NORMED)
}

if choice not in methods:
    print("入力エラー")
    exit()

method_name, method = methods[choice]

# 元画像
img = cv2.imread("face3.jpg")

# テンプレート画像
temp = cv2.imread("./31/template.jpg")

if img is None:
    print("face3.jpg がありません")
    exit()

if temp is None:
    print("template.jpg がありません")
    exit()

# グレースケール変換
gray_img = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

gray_temp = cv2.cvtColor(
    temp,
    cv2.COLOR_BGR2GRAY
)

# テンプレートマッチング
result = cv2.matchTemplate(
    gray_img,
    gray_temp,
    method
)

# 最大値・最小値取得
min_val, max_val, min_pt, max_pt = cv2.minMaxLoc(result)

# SSD・SAD
if method in [
    cv2.TM_SQDIFF,
    cv2.TM_SQDIFF_NORMED
]:
    top_left = min_pt
    score = min_val

# NCC・ZNCC
else:
    top_left = max_pt
    score = max_val

# テンプレートサイズ
h, w = gray_temp.shape

bottom_right = (
    top_left[0] + w,
    top_left[1] + h
)

# 矩形描画
cv2.rectangle(
    img,
    top_left,
    bottom_right,
    (0, 0, 255),
    3
)

print("手法 :", method_name)
print("類似度 :", score)

cv2.imshow("Template", temp)
cv2.imshow("Result", img)

cv2.waitKey(0)
cv2.destroyAllWindows()