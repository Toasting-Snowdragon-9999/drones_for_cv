import cv2
from matplotlib.pyplot import draw
import numpy as np
import os 

path = "python/img/"

class Vision:

    def __init__(self, original_image = "img/original_field.jpg"):
        self.original_image = original_image
        self.img = None

    def get_img(self):
        return self.img

    def load_image(self) -> None:
        self.img = cv2.imread(os.path.join(path, "original_field.jpg"))
        if self.img is None:
            print("Error: Image not found.")

    def show_image(self, img: np.array) -> None:
        if img is None:
            print("Error: No image to display.")
            return

        # Detect shape
        if img.ndim == 2:  # Grayscale
            height, width = img.shape
        elif img.ndim == 3:  # Color
            height, width, _ = img.shape
        else:
            print("Error: Unsupported image shape.")
            return

        # Resize if larger than Full HD
        if height > 1080 and width > 1920:
            img = cv2.resize(img, (width // 4, height // 4))

        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def make_empty_image(self) -> None:
        self.img = np.zeros((100, 200, 3), np.uint8)
        cv2.line(self.img, (20, 30), (40, 120),(0, 0, 255), 3)
        cv2.imwrite(os.path.join(path, "test.png"), self.img)

    def draw_penis(self) -> None:
        temp_img = self.img.copy()
        cv2.rectangle(temp_img, (100, 100), (200, 400), (255, 0, 0), -1)  # Main body
        cv2.circle(temp_img, (100, 100), 50, (255, 0, 0), -1)  # Left testicle
        cv2.circle(temp_img, (200, 100), 50, (255, 0, 0), -1)  # Right testicle
        cv2.circle(temp_img, (150, 400), 50, (255, 0, 0), -1)  # the tip
        cv2.imwrite(os.path.join(path, "[1.6.2] draw_something.png"), temp_img)

    def save_color_channels(self) -> None:
        b, g, r = cv2.split(self.img)
        cv2.imwrite(os.path.join(path, "[1.6.3]blue_channel.png"), b)
        cv2.imwrite(os.path.join(path, "[1.6.3]green_channel.png"), g)
        cv2.imwrite(os.path.join(path, "[1.6.3]red_channel.png"), r)

    def save_upper_half(self) -> None:
        height, width = self.img.shape[:2]
        upper_half = self.img[:height // 2, :]
        cv2.imwrite(os.path.join(path, "[1.6.4]upper_half.png"), upper_half)

    def save_hsv_image(self) -> None:
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        cv2.imwrite(os.path.join(path, "[1.6.5]h_channel.png"), h)
        cv2.imwrite(os.path.join(path, "[1.6.5]s_channel.png"), s)
        cv2.imwrite(os.path.join(path, "[1.6.5]v_channel.png"), v)

    def brightest_pixel(self) -> None:
        if self.img is None:
            print("Error: No image provided.")
            return None

        grey_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        max_val = np.max(grey_img)
        max_coords = np.unravel_index(np.argmax(grey_img), grey_img.shape)

        new_img = self.img.copy()
        # circle the brightest pixel
        cv2.circle(new_img, (max_coords[1], max_coords[0]), 100, (0, 255, 0), 3)
        cv2.circle(new_img, (max_coords[1], max_coords[0]), 10, (0, 255, 0), 1)

        cv2.imwrite(os.path.join(path, "[1.6.6]brightest_pixel.png"), new_img)

    def load_flower_image(self) -> None:
        self.img = cv2.imread(os.path.join(path, "flower.jpg"))
        if self.img is None:
            print("Error: Flower image not found.")
    
    def find_yellow_hsv(self):
        if self.img is None:
            print("Error: No image provided.")
            return None

        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
        yellow_pixels = cv2.bitwise_and(self.img, self.img, mask=mask)

        cv2.imwrite(os.path.join(path, "[1.6.7]yellow_hsv_pixels.png"), yellow_pixels)
        cv2.imwrite(os.path.join(path, "[1.6.7]yellow_hsv_mask.png"), mask)

    def find_yellow_lab(self):
        if self.img is None:
            print("Error: No image provided.")
            return None

        lab_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2Lab)
        lower_yellow = np.array([50, 120, 150])  # L, a, b
        upper_yellow = np.array([255, 150, 255])
        mask = cv2.inRange(lab_img, lower_yellow, upper_yellow)
        yellow_pixels = cv2.bitwise_and(self.img, self.img, mask=mask)

        cv2.imwrite(os.path.join(path, "[1.6.7]yellow_pixels_lab.png"), yellow_pixels)
        cv2.imwrite(os.path.join(path, "[1.6.7]yellow_mask_lab.png"), mask)

    def get_each_pixel_green(self):
        if self.img is None:
            print("Error: No image provided.")
            return None
        height, width, channels = self.img.shape    
        green_pixels = np.zeros((height, width, 3), dtype=np.uint8)
        green_pixels[:, :, 1] = self.img[:, :, 1]
        cv2.imshow("green", green_pixels)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return green_pixels

    def get_binary_mask(self) -> np.array:
        if self.img is None:
            print("Error: No image provided.")
            return None
        
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # Everything except green hues (90–130) and low saturation
        lower_green = np.array([35, 90, 30])   
        upper_green = np.array([55, 255, 255]) 

        # Mask the green
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Invert mask to keep everything except green
        non_green_mask = cv2.bitwise_not(green_mask)

        # Apply mask to original image
        filtered = cv2.bitwise_and(self.img, self.img, mask=non_green_mask)

        # Combine masks
        # mask = cv2.bitwise_or(mask1, mask2)

        gauss_filter = cv2.GaussianBlur(non_green_mask, (5, 5), 0)
        # hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        # lower_bound = np.array([0, 40, 50])    # allow more low-saturation
        # upper_bound = np.array([179, 255, 255])
        # mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        animal_shapes = cv2.bitwise_and(hsv, hsv, mask=non_green_mask)
        
        # h, s, v = cv2.split(hsv_img)
        cv2.imwrite(os.path.join(path, "[1.6.8]Filteren_animals.png"), filtered)
        # _, binary_img = cv2.threshold(s, 0, 200, cv2.THRESH_BINARY)
        self.show_image(non_green_mask)
        self.show_image(gauss_filter)
        # Calculate moments
        return gauss_filter


    def find_contours(self, binary_mask: np.ndarray) -> np.ndarray:
        if binary_mask is None:
            print("Error: No binary mask provided.")
            return []

        # Ensure single-channel uint8
        if binary_mask.ndim == 3:
            binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
        if binary_mask.dtype != np.uint8:
            binary_mask = (binary_mask > 0).astype(np.uint8) * 255

        # Detect edges
        edges = cv2.Canny(binary_mask, 100, 200)
        if not np.any(edges):
            print("No edges found.")
            return edges

        cnts_info = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]

        # Nothing found?
        if not contours:
            print("No contours found.")
            self.show_image(self.img if self.img is not None else edges)
            return edges

        # OPTIONAL: ignore tiny specks
        min_area = 200  # tweak or set to 0 to disable
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        if not contours:
            print("No contours above min_area.")
            self.show_image(self.img if self.img is not None else edges)
            return edges

        # Keep only the largest N contours
        top_k = -1  # change to 3, 5, ... if you want more
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:top_k]

        # Draw on a copy of the original image
        vis = self.img.copy() if self.img is not None else cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 5)

        self.show_image(vis)
        return edges


    def hu_moments(self, contour: list):
        moments = cv2.moments(contour)
        hu_features = cv2.HuMoments(moments).flatten()
        print("Hu Moments:", hu_features)
        return hu_features

    def print_hsv_values(self, pixel_start_location, pixel_end_location) -> None:
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        h_values, s_values, v_values = [], [], []

        for y in range(pixel_start_location[1], pixel_end_location[1] + 1):
            for x in range(pixel_start_location[0], pixel_end_location[0] + 1):
                h, s, v = hsv_img[y, x]
                h_values.append(h)
                s_values.append(s)
                v_values.append(v)
                print(f"Pixel at ({x}, {y}) - H: {h}, S: {s}, V: {v}")

        # Compute statistics
        def stats(values):
            return min(values), max(values), sum(values) / len(values)

        h_min, h_max, h_avg = stats(h_values)
        s_min, s_max, s_avg = stats(s_values)
        v_min, v_max, v_avg = stats(v_values)

        print("\n=== HSV Statistics in Range ===")
        print(f"Hue       -> Min: {h_min}, Max: {h_max}, Avg: {h_avg:.2f}")
        print(f"Saturation-> Min: {s_min}, Max: {s_max}, Avg: {s_avg:.2f}")
        print(f"Value     -> Min: {v_min}, Max: {v_max}, Avg: {v_avg:.2f}")
        print("\n=== HSV Statistics ===")