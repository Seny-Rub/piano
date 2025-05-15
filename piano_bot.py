import cv2
import numpy as np
import mss
import pyautogui
import time
import threading

running = True
flag_allow_click_mouse = False
flag_allow_click_mouse_thread = False
number_of_shots = 6

screen_width, screen_height = pyautogui.size()
roi = (int((screen_width - 1151) / 2), int((screen_height - 646) / 2), 1151, 646)
roi_2 = (151, 413, 704, 222)

x_global_click, y_global_click = 0, 0

from pynput import keyboard as pynput_keyboard
from pynput.keyboard import Controller as KeyboardController, Key

def on_press(key):
    global running, flag_allow_click_mouse
    try:
        if key.char == '1':
            print("Key 1 pressed!")
            running = False
        elif key.char == '2':
            print(f"Key 2 pressed! {flag_allow_click_mouse} Can click mouse")
            flag_allow_click_mouse = not flag_allow_click_mouse
    except AttributeError:
        pass

from pynput.mouse import Button, Controller as MouseController

mouse_controller = MouseController()

def mouse_click(x=None, y=None, button='left'):
    try:
        if x is not None and y is not None:
            mouse_controller.position = (x, y)
        if button == 'left':
            mouse_button = Button.left
        elif button == 'right':
            mouse_button = Button.right
        else:
            raise ValueError("Unsupported button. Use 'left' or 'right'.")
        mouse_controller.click(mouse_button)
    except Exception as e:
        print(f"Error: {e}")

keyboard_controller = KeyboardController()

def press_gap():
    global number_of_shots
    if number_of_shots <= 1:
        print("space")
        keyboard_controller.press(Key.space)
        time.sleep(0.05)
        keyboard_controller.release(Key.space)
        number_of_shots = 6

def create_trackbars():
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, lambda x: None)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, lambda x: None)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, lambda x: None)
    cv2.createTrackbar("H - H", "Trackbars", 0, 179, lambda x: None)
    cv2.createTrackbar("H - S", "Trackbars", 0, 255, lambda x: None)
    cv2.createTrackbar("H - V", "Trackbars", 1, 255, lambda x: None)

def capture_and_filter_roi():
    global number_of_shots, roi, roi_2
    global x_global_click, y_global_click
    global flag_allow_click_mouse_thread
    global flag_allow_click_mouse

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if sum(roi) == 0:
            roi = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)
            print('roi', roi)
            cv2.destroyAllWindows()

        x, y, w, h = roi

        if w == 0 or h == 0:
            print("ROI not selected or has zero size. Program ended.")
            return

        print(f"Selected ROI: x={x}, y={y}, width={w}, height={h}")

        create_trackbars()

        dict_number_last_x_coord = {}
        while running:
            if not cv2.getWindowProperty("Trackbars", cv2.WND_PROP_VISIBLE):
                create_trackbars()

            screenshot = sct.grab({"left": x, "top": y, "width": w, "height": h})
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img_game = img.copy()
            if sum(roi_2) != 0:
                x2, y2, w2, h2 = roi_2
                img = img[y2:y2 + h2, x2:x2 + w2, :]

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lh = cv2.getTrackbarPos("L - H", "Trackbars")
            ls = cv2.getTrackbarPos("L - S", "Trackbars")
            lv = cv2.getTrackbarPos("L - V", "Trackbars")
            hh = cv2.getTrackbarPos("H - H", "Trackbars")
            hs = cv2.getTrackbarPos("H - S", "Trackbars")
            hv = cv2.getTrackbarPos("H - V", "Trackbars")

            lower_bound = np.array([lh, ls, lv])
            upper_bound = np.array([hh, hs, hv])

            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            kernel = np.ones((10, 10), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

            output = img.copy()
            local_object_coord = []
            if num_labels == 0:
                press_gap()
            for i in range(1, num_labels):
                x_comp, y_comp, w_comp, h_comp, area = stats[i]
                if area > 400:
                    cv2.rectangle(output, (x_comp, y_comp), (x_comp + w_comp, y_comp + h_comp), (0, 255, 0), 2)
                    cv2.putText(output, f"ID: {i} {area}", (x_comp, y_comp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    x_roi_2 = int(x_comp + w_comp // 2)
                    y_roi_2 = int(y_comp + h_comp // 2)
                    x_roi = x_roi_2 + roi_2[0]
                    y_roi = y_roi_2 + roi_2[1]
                    x_global = x_roi + roi[0]
                    y_global = y_roi + roi[1]

                    if flag_allow_click_mouse:
                        delta_x = 0
                        if i in dict_number_last_x_coord.keys():
                            delta_x = x_global - dict_number_last_x_coord[i]
                            if delta_x < 0:
                                delta_x = 5
                        mouse_click(x_global + delta_x, y_global, button='left')
                        print("Mouse click")
                        number_of_shots -= 1
                        press_gap()
                        dict_number_last_x_coord[i] = x_global
                    cv2.circle(img_game, (x_roi, y_roi), 5, (255, 0, 0), 3)
                    local_object_coord.append([int(x_comp + w_comp // 2), int(y_comp + h_comp // 2)])

            cv2.imshow("Original ROI", img)
            cv2.imshow("Mask", mask)
            cv2.imshow("Detected Objects", output)
            cv2.imshow("img_game", img_game)

            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            if key == ord('r'):
                roi_2 = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)
                print('roi_2', roi_2)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    listener = pynput_keyboard.Listener(on_press=on_press)
    listener.start()
    capture_and_filter_roi()