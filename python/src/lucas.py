import cv2
import numpy as np


def wait_and_destroy():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hu_log(hu7):
    # v_i = -sign(h_i) * log10(|h_i|)
    hu7 = np.array(hu7, dtype=np.float64).flatten()
    with np.errstate(divide="ignore", invalid="ignore"):
        v = -np.sign(hu7) * np.log10(np.abs(hu7))
    return v


def circularity(area, perimeter):
    if perimeter <= 0:
        return 0.0
    return 4.0 * np.pi * area / (perimeter * perimeter)


def solidity(cnt, area):
    hull = cv2.convexHull(cnt)
    ha = cv2.contourArea(hull)
    if ha <= 0:
        return 0.0
    return float(area) / float(ha)


def passes_shape_bands(hu1, hu2):
    """
    Bands derived from your 4 examples, padded for variance.
    Tweak if you get false positives/negatives.
    """
    # Cluster A (Contour 1-like)
    A_hu1 = (0.25, 0.30)
    A_hu2 = (0.03, 0.06)
    in_A = (A_hu1[0] <= hu1 <= A_hu1[1]) and (A_hu2[0] <= hu2 <= A_hu2[1])

    # Cluster B (Contours 2–4-like)
    B_hu1 = (0.31, 0.36)
    B_hu2 = (0.075, 0.095)
    in_B = (B_hu1[0] <= hu1 <= B_hu1[1]) and (B_hu2[0] <= hu2 <= B_hu2[1])

    return in_A or in_B, ("Animal" if in_A else ("Animal" if in_B else "-"))


def main():
    # Load image
    img = cv2.imread("../images/capture_6/img_1.jpg")
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV range for green
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create mask for green and invert
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    # Optional: clean up specks (helps stability)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, k, iterations=1)

    # Apply mask to remove grass
    result = cv2.bitwise_and(img, img, mask=clean)

    # Grayscale & binary
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area
    min_area = 1000
    max_area = 100000
    contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

    # Sort and keep top N by area
    top_n = 4
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:top_n]

    # Additional priors
    circ_min, circ_max = 0.05, 0.60  # tune as needed
    sol_min = 0.60  # tune as needed
    draw_color_keep = (0, 0, 255)
    draw_color_drop = (0, 255, 255)

    contour_img = img.copy()
    rect_img = img.copy()
    for i, cnt in enumerate(contours, start=1):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Hu moments (linear + log)
        m = cv2.moments(cnt)
        hu = cv2.HuMoments(m).flatten()
        hu_log_vals = hu_log(hu)

        # Simple shape priors
        circ = circularity(area, perimeter)
        sol = solidity(cnt, area)

        # Core relation: (Hu1, Hu2) cluster bands
        in_band, band = passes_shape_bands(hu[0], hu[1])

        # Final keep rule
        keep = in_band and (circ_min <= circ <= circ_max) and (sol >= sol_min)

        print(
            f"\nContour {i}: A={area:.1f}, P={perimeter:.2f}, circ={circ:.3f}, sol={sol:.3f}"
        )
        print("  Hu (linear): " + ", ".join([f"{x:.6e}" for x in hu]))
        print("  Hu (log)   : " + ", ".join([f"{x:.3f}" for x in hu_log_vals]))
        print(f"  Band match : {band}")
        print(f"  -> keep    : {keep}")

        # Draw contour
        cv2.drawContours(
            contour_img, [cnt], -1, draw_color_keep if keep else draw_color_drop, 2
        )

        # Draw bounding rectangle if kept
        if keep:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(rect_img, (x, y), (x + w, y + h), draw_color_keep, 2)
            cv2.putText(
                rect_img,
                f"{band}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                draw_color_keep,
                2,
                cv2.LINE_AA,
            )

    # Show results
    cv2.imshow("Original", img)
    wait_and_destroy()
    cv2.imshow("Without Grass (cleaned)", result)
    wait_and_destroy()
    cv2.imshow("Contours (keep=red, drop=yellow)", contour_img)
    wait_and_destroy()
    cv2.imshow("Animals Highlighted", rect_img)
    wait_and_destroy()


if __name__ == "__main__":
    main()
