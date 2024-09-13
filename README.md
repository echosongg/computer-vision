# computer-vision

## Assignment 1: Harris Corner Detection and Image Transformation
- Implemented **Harris Corner Detection** including the calculation of the corner response function and non-maximum suppression. The `Harris_cornerness` function was designed to compute the corner response, and the `non_max_suppression` function was used to detect corners.
- Compared self-implemented corner detection with OpenCV's `cv2.cornerHarris()` and analyzed the results on rotated images.
- Developed **inverse image transformation** with bilinear interpolation and tested it on rotated images.
- Applied **domain adaptation** using ResNet-34 and optimized the model for Real Domain and Sketch Domain using Maximum Mean Discrepancy (MMD) as the loss function.

## Assignment 2: Canny Edge Detection and Hough Transform for Circle Detection
- Converted RGB images to grayscale and applied **Canny edge detection** with various threshold combinations to optimize for circle detection.
- Implemented **Hough Transform** to detect circles and returned circle candidates in the form of (x, y, radius).
- Developed **Non-Maximum Suppression** to remove duplicate circle detections.
- Superimposed the detected circles on the original images and displayed them.
- Conducted experiments to test different parameters (e.g., radius range, radius increment, theta count, threshold) for the Hough Transform and analyzed their impact on detection accuracy.

## Assignment 3: Image Homography and Panorama Creation
- Implemented **Homography transformation** using Direct Linear Transformation (DLT) and normalized point coordinates to compute a homography matrix.
- Used **SIFT feature descriptors** to find matching points between images and saved the corresponding points for later use.
- Applied the **RANSAC algorithm** to robustly estimate the homography matrix and filter out outliers.
- Used the computed homography to warp one image and stitched it with another to create a panorama.
- Analyzed the effects of different homography parameters and point selection on the accuracy of the rectified results.