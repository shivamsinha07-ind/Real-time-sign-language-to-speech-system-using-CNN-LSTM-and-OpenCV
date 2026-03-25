import cv2
import numpy as np
import os
import settings
CLASSES = settings.CLASSES
ROOT_DATA = settings.ROOT_DATA
FRAMES_PER_SEQ = settings.FRAMES_PER_SEQ
FRAME_SIZE = settings.FRAME_SIZE
TOTAL_SEQUENCES = settings.TOTAL_SEQUENCES  
def detect(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = np.array([2,30,60], dtype=np.uint8)
    high = np.array([25,255,255], dtype=np.uint8)

    mask = cv2.inRange(hsv, low, high)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=3)

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 4000:
        return None

    x,y,w,h = cv2.boundingRect(c)
    pad = 25

    x = max(0,x-pad)
    y = max(0,y-pad)
    w = min(frame.shape[1]-x,w+2*pad)
    h = min(frame.shape[0]-y,h+2*pad)

    crop = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray,(FRAME_SIZE[1],FRAME_SIZE[0]))

for cls in CLASSES:
    for s in range(TOTAL_SEQUENCES):
        os.makedirs(os.path.join(ROOT_DATA,cls,str(s)),exist_ok=True)

cap = cv2.VideoCapture(0)

for cls in CLASSES:
    for seq in range(TOTAL_SEQUENCES):
        for f in range(FRAMES_PER_SEQ):
            ret,frame = cap.read()
            img = detect(frame)

            if img is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(gray,(FRAME_SIZE[1],FRAME_SIZE[0]))

            path = os.path.join(ROOT_DATA,cls,str(seq),f"{f}.jpg")
            cv2.imwrite(path,img)

            cv2.imshow("capture",frame)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break

cap.release()
cv2.destroyAllWindows()