import cv2


def check_video(filename):

    cap = cv2.VideoCapture(filename)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 再生時間を計算
    seconds = frames / fps

    print("ファイル:", filename)
    print("FPS:", fps)
    print("フレーム数:", frames)
    print("再生時間:", seconds, "秒")
    print()


check_video("IMG_9040.MOV")
check_video("anti_shake.mp4")