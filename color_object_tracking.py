# Color object tracking (single target)
# follow a colored game piece and get center/area back to op mode

import cv2, numpy as np

def runPipeline(image, llrobot):
    # llrobot[0:2] can carry HSV tweak from robot (optional)
    h_offset = int(llrobot[0]) if len(llrobot) > 0 else 0
    s_min = int(llrobot[1]) if len(llrobot) > 1 else 70

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Base green range; adjust via llrobot
    lower = (60 + h_offset, s_min, 70)
    upper = (85 + h_offset, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    llpython = [0, 0, 0, 0]   # [hasTarget, cx_px, cy_px, area_px]

    largest = np.array([[]])
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(largest)
        cx, cy = x + w/2, y + h/2
        area = w*h
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.drawContours(image, [largest], -1, (255,0,255), 2)
        llpython = [1, float(cx), float(cy), float(area)]

    return largest, image, llpython
