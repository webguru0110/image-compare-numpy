import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def resize_image(img, target_size=(2640, 1760)):
    """ Resize image to a target size. """
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def reduce_noise(img):
    """ Apply Gaussian blur for noise reduction. """
    # return cv2.GaussianBlur(img, (9, 9), 0) # 26.53%
    return cv2.GaussianBlur(img, (7, 7), 0) # 28.27%
    # return cv2.medianBlur(img, 9)  # 15.08%
    # return cv2.stackBlur(img, (5,5))  # 27%

def rotate_image(img, angle):
    """ Rotate the image by the given angle. """
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def normalize_intensity(img):
    """ Apply histogram equalization to normalize intensity. """
    return cv2.equalizeHist(img)

def align_images(im1, im2):
    """ Align im1 to im2 using ORB feature matching and homography. """

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2, None)

    # Match features using Hamming distance
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    # Extract good matches
    numGoodMatches = int(len(matches) * 0.15)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Warp im1 to align with im2
    height, width = im2.shape[:2]
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg

def preprocess_images(im1, im2):
    """ Preprocess images to align orientation, size, noise level, and intensity. """
    # Step 1: Resize both images to the same size
    im1_resized = resize_image(im1, target_size=(1500, 1500))
    im2_resized = resize_image(im2, target_size=(1500, 1500))

    # Step 2: Reduce noise
    im1_denoised = reduce_noise(im1_resized)
    im2_denoised = reduce_noise(im2_resized)

    # Step 3: Normalize intensity
    im1_normalized = normalize_intensity(im1_denoised)
    im2_normalized = normalize_intensity(im2_denoised)

    # Optional: Rotate image if orientation is known
    # im1_rotated = rotate_image(im1_normalized, angle=0)  # Adjust angle if needed
    # im2_rotated = rotate_image(im2_normalized, angle=0)

    return im1_normalized, im2_normalized

def compare_images(im1, im2):
    """ Compare two images using SSIM after aligning them. """
    # Preprocess both images
    im1_prep, im2_prep = preprocess_images(im1, im2)

    # Align the first image to the second
    im1_aligned = align_images(im1_prep, im2_prep)

    # Compare using SSIM
    score, diff = ssim(im1_aligned, im2_prep, full=True)
    print(f"Similarity Score: {score * 100:.2f}%")

    return score, diff

# Load images (in grayscale for simplicity)
im1 = cv2.imread("ridge_binary.jpg", cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread("binary_craquelures.jpg", cv2.IMREAD_GRAYSCALE)

if im1 is None or im2 is None:
    raise ValueError("One of the images didn't load properly.")

# Compare the two images
score, diff = compare_images(im1, im2)

# Display results
diff = (diff * 255).astype("uint8")
cv2.imshow("Difference", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()
