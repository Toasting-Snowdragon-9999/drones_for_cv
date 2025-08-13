import cv2
from matplotlib.pyplot import draw
import numpy as np
from src.plotter import Plotter
from src.vision import Vision

def show_hsv_picker(img_bgr):
    # Convert to HSV once
    if img_bgr is None:
        return
    height, width, _ = img_bgr.shape
    if height > 1080 or width == 1920:
        img_bgr = cv2.resize(img_bgr, (width // 4, height // 4))
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click
            h, s, v = hsv[y, x]
            print(f"Pixel at ({x}, {y}) - H: {h}, S: {s}, V: {v}")

    cv2.namedWindow("Click to see HSV")
    cv2.setMouseCallback("Click to see HSV", mouse_callback)

    while True:
        cv2.imshow("Click to see HSV", hsv)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break
    cv2.destroyAllWindows()

def main():

    vision = Vision()
    vision.load_image()
    # vision.draw_penis()
    # vision.save_color_channels()
    # vision.save_upper_half()
    # vision.save_hsv_image()
    # vision.brightest_pixel()
    # vision.load_flower_image()
    # vision.find_yellow_hsv()
    # vision.find_yellow_lab()
    # plotter = Plotter()
    # plotter.plot_histogram(vision.get_img())
    # plotter.plot_histogram_g(vision.get_img())
    # vision.get_each_pixel_green()
    vision.find_edge(vision.get_binary_mask())
    # show_hsv_picker(vision.get_img())

if __name__ == "__main__":
    main()
    