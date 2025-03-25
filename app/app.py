import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(title, image):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def crop_rectangle(image, coordinates):
    x1, y1, x2, y2 = coordinates
    cropped = image[y1:y2, x1:x2]
    return cropped

color_drawing = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255)
}

# Load image
relative_path = 'foto/2.jpg'
image_path = os.path.join(os.path.dirname(__file__), relative_path)
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found in path: {image_path}")

display_image("Original Image", image)
height, width = image.shape[:2]

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
display_image("Grayscale", gray)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_gray = clahe.apply(gray)
display_image("CLAHE", clahe_gray)

# Blurring for better edge detection
blurred = cv2.GaussianBlur(clahe_gray, (5, 5), 0)
display_image("Blurred Image", blurred)

# Edge detection using Canny
edges = cv2.Canny(blurred, 44, 75, apertureSize=3, L2gradient=True)
display_image("Edges detected using Canny", edges)

# Performing morphological closing to ensure edge continuity
morphed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)),
                                 iterations=5)

# Finding contours
contours, hierarchy = cv2.findContours(morphed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

# Drawing the largest detected contour on the image
contour_image = image.copy()
for cnt in contours:
    cv2.drawContours(contour_image, [cnt], -1, (0, 255, 0), 5)
display_image("Detected Contour", contour_image)

# Finding vertices of the largest contour
contour = contours[0]
epsilon = 0.02 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)

# Drawing vertices on the image
vertices_image = image.copy()
for point in approx:
    cv2.circle(vertices_image, tuple(point[0]), 10, (0, 255, 0), -1)
display_image("Contour Vertices", vertices_image)

# If there are more than 4 vertices (perspective transform requires exactly 4 points), choose the ones farthest from the contour's center
if len(approx) > 4:
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    distances = [cv2.norm((cx, cy), tuple(point[0])) for point in approx]
    farthest_points = [approx[i] for i in np.argsort(distances)[-4:]]
else:
    farthest_points = approx

# Arranging points in clockwise order for perspective transform
farthest_points = np.array(farthest_points)
center = farthest_points.mean(axis=0)
angles = np.arctan2(farthest_points[:,0,1] - center[0,1], farthest_points[:,0,0] - center[0,0])
farthest_points = farthest_points[np.argsort(angles)]

# Perspective transform
destination_points = np.float32([[0, 0], [width*0.8, 0], [width*0.8, height], [0, height]])
current_points = np.float32([point[0] for point in farthest_points])
perspective_matrix = cv2.getPerspectiveTransform(current_points, destination_points)
warped_image = cv2.warpPerspective(image, perspective_matrix, (int(width*0.8), height))
display_image("Perspective Transformation", warped_image)

conflict_area = crop_rectangle(warped_image, [1668, 1820, 3020, 2580])
display_image("Conflict Area", conflict_area)

# Preparing for edge detection with Canny in HoughCircles
conflict_area_for_circles_detection = conflict_area.copy()
gray_conflict = cv2.cvtColor(conflict_area_for_circles_detection, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_conflict_area_for_circles_detection = clahe.apply(gray_conflict)
blurred_conflict_area_for_circles_detection = cv2.GaussianBlur(clahe_conflict_area_for_circles_detection, (7, 7), 1.5)

# Approximation for performing Canny in HoughCircles
param2 = 100
param1 = param2 / 2
canny_v2 = cv2.Canny(blurred_conflict_area_for_circles_detection, param1, param2, apertureSize=3)
display_image("Before HoughCircles", canny_v2)

# Detecting circles - HoughCircles
circles = cv2.HoughCircles(blurred_conflict_area_for_circles_detection, cv2.HOUGH_GRADIENT_ALT,
                           dp=2, minDist=280, param1=param2, param2=0.5, minRadius=150, maxRadius=250)

conflict_area_for_circles_display = conflict_area.copy()
conflict_area_for_circles_display_2 = conflict_area.copy()

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    # Drawing circles and their centers
    for (x, y, r) in circles:
        cv2.circle(conflict_area_for_circles_display, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(conflict_area_for_circles_display, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        cv2.circle(conflict_area_for_circles_display_2, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(conflict_area_for_circles_display_2, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    display_image("Detected Circles - HoughCircles", conflict_area_for_circles_display)
else:
    raise Exception("HoughCircles: no circles found :(")

# Converting to HSV color space
conflict_area_for_unit_detection = conflict_area.copy()
hsv = cv2.cvtColor(conflict_area_for_unit_detection, cv2.COLOR_BGR2HSV)

# Creating a mask that excludes detected circles
circle_mask = np.ones(conflict_area_for_unit_detection.shape[:2], dtype="uint8") * 255
if circles is not None:
    for (x, y, r) in circles:
        cv2.circle(circle_mask, (x, y), r, 0, -1)

# Displaying circle exclusion mask
display_image("Mask excluding circles", circle_mask)

# Defining color ranges in HSV space
colors = {
    "red": ((140, 33, 86), (220, 112, 187)),
    "green": ((68, 48, 54), (99, 243, 208)),
    "blue": ((100, 50, 50), (135, 255, 255)),
    "yellow": ((14, 35, 116), (20, 122, 224))
}

# Dictionaries for storing the number of contours for each color outside and inside circles
contour_count_outside_circle = {
    "red": 0,
    "green": 0,
    "blue": 0,
    "yellow": 0
}

contour_count_inside_circle = {
    "red": 0,
    "green": 0,
    "blue": 0,
    "yellow": 0
}

# Creating and displaying masks for each color
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
for color, (lower, upper) in colors.items():
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_and(mask, mask, mask=circle_mask)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=3)
    display_image(f"Mask for color {color}", mask)

    # Finding contours of units in the conflict area with a minimum area threshold to exclude unwanted artifacts
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) >= 300:
            contour_count_outside_circle[color] += 1
            cv2.drawContours(conflict_area_for_unit_detection, [cnt], -1, color_drawing[color], 3)
            cv2.drawContours(conflict_area_for_circles_display, [cnt], -1, color_drawing[color], 3)

# Displaying the image with highlighted unit contours in the conflict area
display_image("Image with highlighted unit contours in the conflict area", conflict_area_for_unit_detection)

# Processing each detected circle
conflict_area_for_unit_detection_in_circle = conflict_area.copy()
for (x, y, r) in circles:
    circle_mask_local = np.zeros_like(circle_mask)
    cv2.circle(circle_mask_local, (x, y), r, 255, -1)
    hsv_circle = cv2.bitwise_and(hsv, hsv, mask=circle_mask_local)
    display_image(f"Circle {x}, {y}", hsv_circle)

    # Creating and displaying masks for each color
    for color, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv_circle, lower, upper)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Finding contours of units inside circles
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) >= 100:
                contour_count_inside_circle[color] += 1
                cv2.drawContours(conflict_area_for_circles_display, [cnt], -1, color_drawing[color], 3)
                cv2.drawContours(conflict_area_for_circles_display_2, [cnt], -1, color_drawing[color], 3)

# Displaying the image with highlighted unit contours in the garrisons
display_image("Image with highlighted unit contours in the garrisons", conflict_area_for_circles_display_2)

# Displaying the image with detected units in garrisons and in the conflict area
display_image("Summary - Image with detected units in garrisons and in the conflict area", conflict_area_for_circles_display)

print()
print("SUMMARY:")
for color in contour_count_outside_circle:
    print(f"Color: {color}")
    print(f"Number of units in conflict area: {contour_count_outside_circle[color]}")
    print(f"Number of units in garrison: {contour_count_inside_circle[color]}")
    print()
