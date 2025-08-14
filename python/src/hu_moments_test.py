import cv2
import numpy as np


def wait_and_destroy():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # Load image
    img = cv2.imread("../images/capture_savannah/img_1.jpg")

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV range for green
    lower_green = np.array([35, 40, 10])  # Hue ~35–85 for green
    upper_green = np.array([85, 255, 255])

    # Create mask for green
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Invert mask to keep everything BUT grass
    mask_inv = cv2.bitwise_not(mask)

    # Apply mask to remove grass
    result = cv2.bitwise_and(img, img, mask=mask_inv)

    # Convert to grayscale
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area
    min_area = 200
    max_area = 500000
    contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

    # Sort and keep top N contours
    top_n = 10
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:top_n]

    # Draw and print stats
    contour_img = img.copy()
    for i, cnt in enumerate(contours, start=1):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Calculate Hu Moments
        moments = cv2.moments(cnt)
        hu_moments = cv2.HuMoments(moments).flatten()

        print(f"\nContour {i}:")
        print(f"  Area      = {area:.2f}")
        print(f"  Perimeter = {perimeter:.2f}")
        print("  Hu Moments:")
        for j, hu in enumerate(hu_moments, start=1):
            print(f"    Hu{j} = {hu:.6e}")

        # Draw contour
        cv2.drawContours(contour_img, [cnt], -1, (0, 0, 255), 2)

    # Resize images
    contour_img = cv2.resize(contour_img, (1500, 1000))
    result = cv2.resize(result, (1500, 1000))
    img = cv2.resize(img, (1500, 1000))

    # Show results
    cv2.imshow("Original", img)
    wait_and_destroy()
    cv2.imshow("Without Grass", result)
    wait_and_destroy()
    cv2.imshow("Filtered Contours", contour_img)
    wait_and_destroy()


if __name__ == "__main__":
    main()
