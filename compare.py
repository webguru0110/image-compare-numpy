import cv2
import numpy as np
from skimage.metrics import structural_similarity

IMAGE_PATH_1 = "ridge_binary.jpg"
IMAGE_PATH_2 = "binary_craquelures.jpg"

# first = cv2.imread(IMAGE_PATH_1)
# second = cv2.imread(IMAGE_PATH_2)

def align_images(im1, im2):
    # Since images are already loaded in grayscale, no need to convert again

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2, None)

    # Match features using Hamming distance
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * 0.15)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Customize match drawing with thicker lines and larger keypoints
    draw_params = dict(matchColor=(0, 255, 0),  # Green color for matches
                    singlePointColor=(255, 0, 0),  # Red color for keypoints
                    matchesMask=None,  # Draw all matches
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Draw the matches with bolder lines and larger keypoints
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None, **draw_params)

    # Draw bolder lines and larger dots by drawing over the default keypoints (optional)
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        pt1 = tuple(map(int, keypoints1[img1_idx].pt))
        pt2 = tuple(map(int, (keypoints2[img2_idx].pt[0] + im1.shape[1], keypoints2[img2_idx].pt[1])))
        cv2.line(imMatches, pt1, pt2, (0, 255, 0), thickness=4)  # Thicker green lines for matches
        cv2.circle(imMatches, pt1, radius=10, color=(0, 0, 255), thickness=3)  # Bigger red dots on keypoints
        cv2.circle(imMatches, pt2, radius=10, color=(0, 0, 255), thickness=3)

    cv2.imshow('matches_bold', imMatches)
    # Save the result
    cv2.imwrite("matches_bold.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography to warp perspective of the first image
    height, width = im2.shape[:2]
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, im2

# Load images
im1 = cv2.imread(IMAGE_PATH_1, cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread(IMAGE_PATH_2, cv2.IMREAD_GRAYSCALE)



def preprocess_image(img):
    """Load an image, convert to grayscale, resize, and apply Gaussian blur."""
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (2640, 1760))  # Resize to common size
    img = cv2.GaussianBlur(img, (31, 31), 0)  # Noise reduction
    cv2.imshow('blur', img)

    return img

# # Convert images to grayscale
# first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
# second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

# Preprocess both images
img1 = preprocess_image(im1)
img2 = preprocess_image(im2)

# Align images
img_1, img_2 = align_images(img1, img2)

# Compute SSIM between two images
score, diff = structural_similarity(img1, img2, full=True)
print("Similarity Score: {:.3f}%".format(score * 100))

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type so we must convert the array
# to 8-bit unsigned integers in the range [0,255] before we can use it with OpenCV
diff = (diff * 255).astype("uint8")

cv2.imwrite("diff.jpg", diff)


# Threshold the difference image, followed by finding contours to
# obtain the regions that differ between the two images
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

# Highlight differences
mask = np.zeros(img1.shape, dtype='uint8')
filled = img2.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area > 100:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img1, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, (0,255,0), -1)
        cv2.drawContours(filled, [c], 0, (0,255,0), -1)

cv2.imshow('first', img1)
cv2.imshow('second', img2)
cv2.imshow('diff', diff)
cv2.imshow('mask', mask)
cv2.imshow('filled', filled)
cv2.waitKey()
