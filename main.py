import cv2
import numpy as np
import pyautogui as pt
import time


class Clicker:
    def __init__(self, target_img_path, speed):
        self.target_img = cv2.imread(target_img_path)
        self.speed = speed
        self.buffer = 15
        self.button_coords = self.find_button_position(self.target_img)

    def find_button_position(self, image, show=False):
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

    def navigate(self):
        position = self.button_coords
        if position is not None:
            pt.click(position[0]+self.buffer, position[1]+self.buffer)
            print("clicked!!!!", position)
        else:
            print("In progress!!")


if __name__ == "__main__":
    # image = cv2.imread("test_images/meet.png")
    # cv2.imshow("screen", image)
    # cv2.waitKey()

    click_obj = Clicker("test_images/meet.png", speed=1)
    time.sleep(5)
    maxTries = 100
    while maxTries:
        if click_obj.navigate() == 0:
            maxTries -= 1

    # cv2.destroyAllWindows()
