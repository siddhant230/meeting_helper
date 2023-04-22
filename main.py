import cv2
import numpy as np
import pyautogui as pt
import time
import random, sys
from copy import deepcopy

class Clicker:
    def __init__(self, target_img_path, speed):
        self.target_img = cv2.imread(target_img_path)
        self.speed = speed
        self.buffer = 5
        self.button_coords = self.find_button_position()
        self.buffer_people = list(range(10, 20)) + list(range(10, 1, -1))
        self.original_num_people = self.get_num_people() + 1

    def find_button_position(self, show=False):
        image = pt.screenshot()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        result = image.copy()
        lower = np.array([155, 25, 0])
        upper = np.array([179, 255, 255])
        mask = cv2.inRange(image, lower, upper)
        result = cv2.bitwise_and(result, result, mask=mask)
        ret, thresh = cv2.threshold(mask, 40, 255, 0)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        bboxes = [cv2.boundingRect(i) for i in contours]
        bboxes = sorted(bboxes, key=lambda x: x[1])
        bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
        x, y, w, h = bboxes[0]
        if show:
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), -1)
            cv2.imshow("res", result)
        return (x, y, x+w, y+h)
    
    def get_contour_only(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,3,1)
        contours, _ = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        mask = np.zeros(img.shape[:-1],np.uint8)
        mask = cv2.drawContours(mask,contours,-1,(255,255,255),-1)
        contours, _ = cv2.findContours(mask, 
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
        return mask
    
    def get_num_people(self):
        image = pt.screenshot()
        image = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        image = image[200:-60, ]
        orig = deepcopy(image)
        image = self.get_contour_only(image)
        # cv2.imshow("processed", image)
        cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        counter = 0
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            area = cv2.contourArea(c)
            if len(approx) > 5 and (area > 2500 and area < 500000):
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(orig, (int(x), int(y)), (int(x+w), int(y+h)), (0,0,255), -1)
                counter += 1
        cv2.imshow("people", orig)
        cv2.waitKey(0)
        return counter

    
    def check_and_click(self):
        curr_num_people = self.get_num_people()
        position = self.button_coords
        print(self.original_num_people, curr_num_people)
        if curr_num_people <= self.original_num_people*0.5 or True:
            pt.click(position[0]+self.buffer, position[1]+self.buffer)
            print("clicked!!!!", position)
            return False
        else:
            print("In progress!!")
            return True


if __name__ == "__main__":
    # image = cv2.imread("test_images/meet.png")
    # cv2.imshow("screen", image)
    # cv2.waitKey()
    time.sleep(1)
    click_obj = Clicker("test_images/meet.png", speed=2)
    # cv2.waitKey(0)
    meeting_ongoing = True
    while meeting_ongoing:
        meeting_ongoing = click_obj.check_and_click()
