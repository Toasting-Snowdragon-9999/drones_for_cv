import matplotlib.pyplot as plt
import numpy as np 
import cv2

class Plotter:
    def __init__(self):
        pass

    def plot_image(self, img: np.ndarray) -> None:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def plot_histogram(self, img: np.ndarray) -> None:
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
        plt.xlim([0, 256])
        plt.show()
    
    def plot_histogram_g(self, img: np.ndarray) -> None:
        hist = cv2.calcHist([img], [1], None, [256], [0, 256])  # [1] instead of 1
        plt.plot(hist, color='g')
        plt.xlim([0, 256])
        plt.show()
    
    def plot_histogram_b(self, img: np.ndarray) -> None:
        color = ('b', 'g', 'r')

        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist, color='b')
        plt.xlim([0, 256])
        plt.show()
    
    def plot_histogram_r(self, img: np.ndarray) -> None:
        color = ('b', 'g', 'r')

        hist = cv2.calcHist([img], [2], None, [256], [0, 256])
        plt.plot(hist, color='r')
        plt.xlim([0, 256])
        plt.show()