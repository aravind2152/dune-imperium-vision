# Eyes of the Empire: A Machine Vision System for Dune Imperium

---

> 🐍🎲 Python-powered Computer Vision project automating gameplay for the board game Dune Imperium 🚀! The system accurately identifies and counts player units—colored cubes—in conflict zones ⚔️ and garrisons 🛡️ using image processing techniques! 🔍✨

---

![duneimperiumvision](https://github.com/user-attachments/assets/88b67c34-bc5c-47ce-9639-871da0651111)

## PROJECT OVERVIEW

This project introduces an automated vision-based solution designed to enhance gameplay for the board game "Dune Imperium" The crucial gameplay element involves resolving military conflicts where players deploy units, represented by colored wooden cubes, into conflict areas. The primary goal of this computer vision system is to identify the units based on their colors, determine their quantity, and categorize them into units actively participating in conflicts or stationed in garrisons around the conflict zone.

The system processes an image of the game board taken from various perspectives and standard lighting conditions, demonstrating flexibility and robustness in practical gaming scenarios.

## MOTIVATION

The project's main motivation is to simplify and automate the tedious task of manually counting units during gameplay. By utilizing a computer vision-based system, players can quickly obtain accurate counts of their units both within and outside the conflict zones, speeding up gameplay and reducing human error. Moreover, the developed solution opens possibilities for expanding into a comprehensive system for monitoring and managing broader game dynamics.

## IMPLEMENTATION

The algorithm follows these main steps:

1. **Image Acquisition**: Captures the entire board image from various angles and lighting conditions.
2. **Grayscale Conversion**: Converts the captured image to grayscale for simplification.
3. **CLAHE Application**: Improves contrast using Contrast Limited Adaptive Histogram Equalization.
4. **Gaussian Blur**: Applies blur to minimize noise and enhance edge detection.
5. **Edge Detection (Canny)**: Identifies significant edges in the image.
6. **Morphological Operations**: Performs morphological closing to ensure edge continuity.
7. **Contour Detection**: Detects and sorts contours by size, isolating the largest one.
8. **Perspective Transformation**: Corrects image perspective using the largest contour's vertices.
9. **Region Cropping**: Extracts the conflict area from the transformed image.
10. **Circle Detection (Hough Transform)**: Detects circles representing garrison locations.
11. **HSV Color Segmentation**: Converts to HSV color space and applies color segmentation to classify units by color.
12. **Unit Counting**: Counts unit contours inside and outside of detected circles, categorizing active and garrisoned units accordingly.

## TECH STACK

- Python 3.8 or higher
- OpenCV
- NumPy
- Matplotlib

## FUTURE ENHANCEMENTS

Possible expansions include:

- Extending the system to track complete gameplay.
- Enhancing the robustness against diverse lighting conditions and image angles.
- Integrating real-time processing to further improve usability.







