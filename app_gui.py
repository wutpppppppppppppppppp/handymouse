from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import HandModule as htm
import time
import autopy
import pyautogui as pag


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
         # Constants for the video capture and frame reduction
        self.wCam, self.hCam = 640, 480
        self.frameR = 100  # Frame Reduction
        self.smoothening = 7
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0


    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        cap.set(3, self.wCam)
        cap.set(4, self.hCam)
        # Initialize the hand detector object, screen size, and click flag
        detector = htm.handDetector(maxHands=2, detectionCon=0.5, trackCon=0.8)
        wScr, hScr = autopy.screen.size()
        clickFlag = 0
        while self._run_flag:
            ret, cv_img = cap.read()  
            cv_img = cv2.flip(cv_img, 1)
            if ret:
                img = detector.findHands(cv_img)
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
                    x3 = np.interp(x1, (self.frameR, self.wCam - self.frameR), (0, wScr))
                    y3 = np.interp(y1, (self.frameR, self.hCam - self.frameR), (0, hScr))
                    # print(x3, y3)

                    # 6. Smoothen Values
                    clocX = self.plocX + (x3 - self.plocX) / self.smoothening
                    clocY = self.plocY + (y3 - self.plocY) / self.smoothening
                    # print(clocX)

                    # 7. Move Mouse by autopy
                    # print("moving mode")
                    cv2.putText(img, "Moving", (45, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 252, 0), 2)
                    autopy.mouse.move(clocX, clocY)
                    #  cv2.circle(img, (x1, y1), 15, (255, 36, 15), cv2.FILLED)
                    self.plocX, self.plocY = clocX, clocY
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
                
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("*")
        self.disply_width = 640
        self.display_height = 480
        
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('Put Your Hands up')

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())