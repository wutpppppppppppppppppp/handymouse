import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.mpStyle = mp.solutions.drawing_styles
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS, self.mpStyle.get_default_hand_landmarks_style(), self.mpStyle.get_default_hand_connections_style())
        return img

    def findPosition(self, img, draw=True):
        xList = []
        yList = []
        # bbox = []
        self.lmListL = []
        self.lmListR = []
        if self.results.multi_hand_landmarks:
            for hand_no, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                # print("hand index:", hand_no,"hand name from handedness:", self.results.multi_handedness[hand_no].classification[0].label, "hand index from handedness: ", self.results.multi_handedness[hand_no].classification[0].index)  # first hand of detection will be labelled as index 0
                # for i in range(21):  # for loop i as a index for each landmark points
                #     print(self.mpHands.HandLandmark(i).name, self.mpHands.HandLandmark(i).value) # from wrist to pinky_tip
                #     print(f'{hand_landmarks.landmark[self.mpHands.HandLandmark(i).value]}')
                # myHand = self.results.multi_hand_landmarks[0]
                for id, lm in enumerate(hand_landmarks.landmark):
                    # print(self.results.multi_handedness[0].classification[0].label, "id is", id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    # print(id, cx, cy)
                    hand = self.results.multi_handedness[0].classification[0].index
                    if (self.results.multi_handedness[hand_no].classification[0].label == 'Left'):
                        self.lmListL.append([id, cx, cy])
                        # print("get left")
                    if (self.results.multi_handedness[hand_no].classification[0].label == 'Right'):
                        self.lmListR.append([id, cx, cy])
                        # print("get right")
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 64, 35), cv2.FILLED)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax
                # if draw:
                #     cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmListL, self.lmListR

    def fingersUp(self):
        fingersL = []
        fingersR = []
        # thumb
        if len(self.lmListL):
            if self.lmListL[self.tipIds[0]][1] < self.lmListL[self.tipIds[0] - 1][1]:
                fingersL.append(1)
            else:
                fingersL.append(0)
            # Fingers
            for id in range(1, 5):
                if self.lmListL[self.tipIds[id]][2] < self.lmListL[self.tipIds[id] - 2][2]:
                    fingersL.append(1)
                else:
                    fingersL.append(0)
        else:
            fingersL = [0, 0, 0, 0, 0]

        # thumb
        if len(self.lmListR):
            if self.lmListR[self.tipIds[0]][1] > self.lmListR[self.tipIds[0] - 1][1]:
                fingersR.append(1)
            else:
                fingersR.append(0)
            # Fingers
            for id in range(1, 5):
                if self.lmListR[self.tipIds[id]][2] < self.lmListR[self.tipIds[id] - 2][2]:
                    fingersR.append(1)
                else:
                    fingersR.append(0)
        else:
            fingersR = [0, 0, 0, 0, 0]
        return fingersL, fingersR

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3, re=255, g=0, b=255):

        x1, y1 = self.lmListL[p1][1:]
        x2, y2 = self.lmListL[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # if draw:
        # cv2.line(img, (x1, y1), (x2, y2), (re, g, b), t)
        # cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
        # cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
        # cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def findRatio(self, p1, p2, img, draw=True, r=15, t=3, re=255, g=0, b=255):

        x1, y1 = self.lmListL[p1][1:]
        x2, y2 = self.lmListL[p2][1:]
        wx, wy = self.lmListL[0][1:]
        # print("x1, y1 = ",self.lmListL[p1][1:])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # from wrist to p1
        wristp1 = math.hypot (x1 - wx, y1 -wy)
        # from p1 to p2
        pp12 = math.hypot(x2 - x1, y2 - y1)
        length = pp12/wristp1
        # if draw:
        # cv2.line(img, (x1, y1), (x2, y2), (re, g, b), t)
        # cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
        # cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
        # cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmListL, lmListR = detector.findPosition(img, draw=False)
        # length, img, lineInfo = detector.findDistance(8, 12, img, re=255, g=208, b=42)
        length, img, lineInfo = detector.findRatio(8, 12, img, re=255, g=208, b=42)

        # if len(lmList) != 0:
        #     print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        esc = cv2.waitKey(1) & 0xff #  if pressed esc
        if esc == 27:
            break
cv2.destroyAllWindows()

if __name__ == "__main__":
    main()