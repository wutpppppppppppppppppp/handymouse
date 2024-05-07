import cv2
import mediapipe as mp
import time
import math

class handDetector():
    """
    A class that uses MediaPipe's Hands solution to detect hands in an image or video frame, and provides methods for finding the position, distance, and ratio of the hands.

    Args:
        mode (bool, optional): Whether to run the solution in real-time or not. Defaults to False.
        maxHands (int, optional): The maximum number of hands to detect. Defaults to 2.
        modelComplexity (int, optional): The model complexity to use. Defaults to 1.
        detectionCon (float, optional): The confidence threshold for hand detection. Defaults to 0.5.
        trackCon (float, optional): The confidence threshold for hand tracking. Defaults to 0.5.

    Attributes:
        mode (bool): Whether the solution is running in real-time or not.
        maxHands (int): The maximum number of hands to detect.
        modelComplexity (int): The model complexity to use.
        detectionCon (float): The confidence threshold for hand detection.
        trackCon (float): The confidence threshold for hand tracking.
        mpHands (mp.solutions.hands): A MediaPipe Hands solution.
        hands (mp.solutions.hands.Hands): A MediaPipe Hands solution instance.
        mpDraw (mp.solutions.drawing_utils): A MediaPipe Drawing solution.
        mpStyle (mp.solutions.drawing_styles): A MediaPipe Drawing styles solution.
        tipIds (list): The indices of the fingers and thumb tips in the HandLandmark list.

    """
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
        """
        Detects hands in an image or video frame.

        Args:
            img (ndarray): The image or video frame to process.
            draw (bool, optional): Whether to draw the hand landmarks on the image or not. Defaults to True.

        Returns:
            ndarray: The processed image or video frame.

        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS, self.mpStyle.get_default_hand_landmarks_style(), self.mpStyle.get_default_hand_connections_style())
        return img

    def findPosition(self, img, draw=True):
        """
        Finds the position of the hands in an image or video frame.

        Args:
            img (ndarray): The image or video frame to process.
            draw (bool, optional): Whether to draw the hand landmarks on the image or not. Defaults to True.

        Returns:
            tuple: A tuple containing the left hand landmarks and the right hand landmarks, as lists of tuples.

        """
        xList = []
        yList = []
        # add up the z axis (far and near the camera)
        # zList = []
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
                    # add up the cz
                    # cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * -1)
                    xList.append(cx)
                    yList.append(cy)
                    # zList.append(cz)
                    
                    # print(id, cx, cy)
                    hand = self.results.multi_handedness[0].classification[0].index
                    if (self.results.multi_handedness[hand_no].classification[0].label == 'Left'):
                        self.lmListL.append([id, cx, cy])
                        # self.lmListL.append([id, cx, cy,cz])
                        # print("get left")
                    if (self.results.multi_handedness[hand_no].classification[0].label == 'Right'):
                        self.lmListR.append([id, cx, cy])
                        # self.lmListR.append([id, cx, cy,cz])
                        # print("get right")
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 64, 35), cv2.FILLED)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                # add the zmin,zmax
                # zmin,zmax = min(zList),max(zList)
                bbox = xmin, ymin, xmax, ymax
                # if draw:
                #     cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmListL, self.lmListR

    def fingersUp(self):
        """
        Determines whether the fingers of the hands are raised or not.

        Returns:
            tuple: A tuple containing the left hand fingers and the right hand fingers, as lists of booleans.

        """
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
        """
        Calculates the distance between two points on the hand.

        Args:
            p1 (int): The index of the first landmark point.
            p2 (int): The index of the second landmark point.
            img (ndarray): The image or video frame to process.
            draw (bool, optional): Whether to draw the hand landmarks on the image or not. Defaults to True.
            r (int, optional): The radius of the circles drawn around the landmark points. Defaults to 15.
            t (int, optional): The thickness of the lines drawn between the landmark points. Defaults to 3.
            re (int, optional): The red color component of the line and circle colors. Defaults to 255.
            g (int, optional): The green color component of the line and circle colors. Defaults to 0.
            b (int, optional): The blue color component of the line and circle colors. Defaults to 255.

        Returns:
            tuple: A tuple containing the calculated distance, the processed image or video frame, and a list of coordinates of the landmark points.

        """
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
        """
        Calculates the ratio between two landmarks on the hand.

        Args:
            p1 (int): The index of the first landmark point.
            p2 (int): The index of the second landmark point.
            img (ndarray): The image or video frame to process.
            draw (bool, optional): Whether to draw the hand landmarks on the image or not. Defaults to True.
            r (int, optional): The radius of the circles drawn around the landmark points. Defaults to 15.
            t (int, optional): The thickness of the lines drawn between the landmark points. Defaults to 3.
            re (int, optional): The red color component of the line and circle colors. Defaults to 255.
            g (int, optional): The green color component of the line and circle colors. Defaults to 0.
            b (int, optional): The blue color component of the line and circle colors. Defaults to 255.

        Returns:
            tuple: A tuple containing the calculated ratio, the processed image or video frame, and a list of coordinates of the landmark points.

        """
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
    
    def findDistanceBetweenHands(self, img, p1_left=8, p2_left=12, p1_right=8, p2_right=12, draw=True, r=15, t=3, re=255, g=0, b=255):
        """
        Calculates the distance between two hands based on specified landmarks.

        Args:
            img (ndarray): The image or video frame to process.
            p1_left (int): The index of the first landmark point on the left hand. Defaults to 8.
            p2_left (int): The index of the second landmark point on the left hand. Defaults to 12.
            p1_right (int): The index of the first landmark point on the right hand. Defaults to 8.
            p2_right (int): The index of the second landmark point on the right hand. Defaults to 12.
            draw (bool): Whether to draw the hand landmarks on the image or not. Defaults to True.
            r (int): The radius of the circles drawn around the landmark points. Defaults to 15.
            t (int): The thickness of the lines drawn between the landmark points. Defaults to 3.
            re (int): The red color component of the line and circle colors. Defaults to 255.
            g (int): The green color component of the line and circle colors. Defaults to 0.
            b (int): The blue color component of the line and circle colors. Defaults to 255.

        Returns:
            tuple: A tuple containing the calculated distance between the two hands and the processed image or video frame. If both hands are not detected, None and the image are returned.
        """

        if self.lmListL and self.lmListR:  # Check if both hands are detected
            x1_left, y1_left = self.lmListL[p1_left][1:]
            x2_left, y2_left = self.lmListL[p2_left][1:]
            x1_right, y1_right = self.lmListR[p1_right][1:]
            x2_right, y2_right = self.lmListR[p2_right][1:]

            # Calculate midpoints of each hand's line
            cx_left, cy_left = (x1_left + x2_left) // 2, (y1_left + y2_left) // 2
            cx_right, cy_right = (x1_right + x2_right) // 2, (y1_right + y2_right) // 2

            # Calculate distance between the midpoints
            distance = math.hypot(cx_right - cx_left, cy_right - cy_left)

            # if draw:
            #     # Draw lines and circles for visualization
            #     cv2.line(img, (x1_left, y1_left), (x2_left, y2_left), (re, g, b), t)
            #     cv2.line(img, (x1_right, y1_right), (x2_right, y2_right), (re, g, b), t)
            #     cv2.circle(img, (cx_left, cy_left), r, (255, 0, 255), cv2.FILLED)
            #     cv2.circle(img, (cx_right, cy_right), r, (255, 0, 255), cv2.FILLED)

            return distance, img

        else:
            return None, img  # Return None if both hands are not detected

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