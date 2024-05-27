import cv2
import numpy as np
import HandModule as htm
import time
import autopy
import pyautogui as pag


# Constants for the video capture and frame reduction
wCam, hCam = 640, 480 # wCam, hCam = 1280, 720 but reduce fps
frameR = 100  # Frame Reduction
smoothening = 7

# Initialize variables for mouse movement and clicks
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Initialize the video capture object
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize the hand detector object, screen size amd click flag
detector = htm.handDetector(maxHands=2, detectionCon=0.5,trackCon=0.8)
wScr, hScr = autopy.screen.size()  # 1536.0 864.0
clickFlag = 0

# Main loop to continuously process frames
while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmListL, lmListR = detector.findPosition(img,draw=False)

    # 2. Get the tip of the index
    if len(lmListR) != 0:
        x1, y1 = lmListR[8][1:] # fix this with line 56,60 in startseperate.py
        # print(x1, y1)

    # 3. Check which fingers are up
    fingersL, fingersR = detector.fingersUp()
    print(fingersL, fingersR)  # [0, 0, 0, 0, 0]

    distance, img = detector.findDistanceBetweenHands(img)
    # print("Distance between hands:", distance)

    if fingersR[1] == 1 and fingersR[2:] == [0, 0, 0]: # moving mode

        # 5. Convert Coordinates
        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
        # print(x3, y3)

        # 6. Smoothen Values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening
        # print(clocX)

        # 7. Move Mouse by autopy
        # print("moving mode")
        cv2.putText(img, "Moving", (45, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 252, 0), 2)
        autopy.mouse.move(clocX, clocY)
        #  cv2.circle(img, (x1, y1), 15, (255, 36, 15), cv2.FILLED)
        plocX, plocY = clocX, clocY

        # 8. Both Index and middle fingers are up : Clicking Left Mode
        # if fingersL[0] == 1 and fingersL[1] == 1 and fingersL[2] == 1 and fingersL[3] == 0 and fingersL[4] == 0:
        if fingersL == [1, 1, 1, 0, 0]:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findRatio(8, 12, img, re=255, g=208, b=42)
            # print(lineInfo)
            print("Left Click : its length is ", length)
            cv2.putText(img, "Left Click", (45, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

            # 10. Click mouse if distance short
            if length < 0.15 and clickFlag == 0:
                #  cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (25, 65), 10, (0, 255, 127), cv2.FILLED)
                pag.mouseDown(button='left')
                clickFlag = 1
            elif length > 0.25 and clickFlag == 1:
                cv2.circle(img, (25, 65), 10, (200, 0, 100), cv2.FILLED)
                pag.mouseUp()
                clickFlag = 0

        # 9. Thumb, index, and middle fingers are up : right click
        # if fingersL[0] == 0 and fingersL[1] == 1 and fingersL[2] == 1 and fingersL[3] == 0 and fingersL[4] == 0:
        if fingersL == [0, 1, 1, 0, 0]:
            # 9.1. Find distance between fingers
            length, img, lineInfo = detector.findRatio(8, 12, img, re=255, g=208, b=42)
            #print("Right Click : its length is ", length)
            cv2.putText(img, "Right Click", (45, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            # 9.2. Click mouse if distance short
            if length < 0.15 and clickFlag == 0:
                #  cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (25, 65), 8, (100, 0, 200), cv2.FILLED)
                pag.mouseDown(button='right')
                clickFlag = 1
            elif length > 0.25 and clickFlag == 1:
                cv2.circle(img, (25, 65), 8, (200, 0, 100), cv2.FILLED)
                pag.mouseUp(button='right')
                clickFlag = 0

        # 10. index, middle, and ring fingers are up : middle click
        if fingersL[0:4] == [1,1,1,1,0]:
            # 10.1. Find distance between fingers
            length, img, lineInfo = detector.findRatio(8, 12, img, re=255, g=208, b=42)
            #   print("Middle click : both length are ", length)
            cv2.putText(img, "Middle Click", (45, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

            # 10.2. Click mouse if distance short
            if length < 0.15 and clickFlag == 0:
                #  cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (25, 65), 8, (100, 0, 200), cv2.FILLED)
                pag.mouseDown(button='middle')
                clickFlag = 1
            elif length > 0.25 and clickFlag == 1:
                cv2.circle(img, (25, 65), 8, (200, 0, 100), cv2.FILLED)
                pag.mouseUp(button='middle')
                clickFlag = 0

    elif fingersR[1:3] == [1, 1] and not any(fingersR[3:]):
        cv2.putText(img, "Scrolling", (45, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 252, 0), 2)
        if fingersL[1] == 1 and fingersL[2] == 1 and fingersL[3] == 0 and fingersL[4] == 0:
            pag.scroll(30)
            cv2.putText(img, "UP", (45, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        elif fingersL[1] == 0 and fingersL[2] == 0 and fingersL[3] == 0 and fingersL[4] == 0:
            cv2.putText(img, "DOWN", (45, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            pag.scroll(-30)

    elif all(fingersR[1:]) and not fingersR[0]:
        pag.mouseUp(button='left')
        pag.mouseUp(button='right')
        pag.mouseUp(button='middle')
    # cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)


    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    
    # 12. Display
    cv2.imshow("Image", img)
    esc = cv2.waitKey(1) & 0xff #  if pressed esc
    if esc == 27:
        pag.mouseUp(button='left')
        pag.mouseUp(button='right')
        pag.mouseUp(button='middle')
        break
cv2.destroyAllWindows()

