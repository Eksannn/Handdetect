import cv2
import mediapipe as mp

# tangan
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.75,
                      min_tracking_confidence=0.75)
mpDraw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# fungsi cek status jari
def finger_status(lm):
    fingers = []
    fingers.append(1 if lm[4].x < lm[3].x else 0)

    tips = [8, 12, 16, 20]
    for tip in tips:
        fingers.append(1 if lm[tip].y < lm[tip-2].y else 0)

    return fingers  # jari

# ganti ganti aja bos
def detect_gesture(fingers):
    if fingers == [1,1,1,1,1]:
        return "HI"
    
    if fingers == [1,0,0,0,0]:
        return "Aku"
    
    if fingers == [0,1,0,0,0]:
        return "Mau"
    
    if fingers == [0,1,0,0,1]:
        return "Suka"
    
    if fingers == [0,1,1,0,0]:
        return "Kamu"

    return None

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        lm = handLms.landmark

        fingers = finger_status(lm)
        gesture = detect_gesture(fingers)

        if gesture:
            cv2.putText(img, gesture, (50,100), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0,0,255), 3)

        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

