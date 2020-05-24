import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image

## ===================================================================================================================
## Author: Su Ye Shan Daniel
## Created Date: 1 May 2020
## Description: An automated application for image quality enhancement by using image registration and median stacking
## ===================================================================================================================

# Declare array and command
cv2.ocl.setUseOpenCL(False)  # this line is important, this will enable ORB function properly in opencv
imgAligned = []
stacked = []
images = []

# Declare sharpening kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
kernel9 = np.array([[-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]])

##set parameter

GOOD_MATCH_PERCENT = 0.15


# Define alignImage
def alignImages(im1, im2):
    # Convert images to grayscale
    G1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    G2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(nfeatures=5000, scoreType=cv2.ORB_HARRIS_SCORE)
    keypoints1, descriptors1 = orb.detectAndCompute(G1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(G2, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # option to save matching image into jpg file:
    # cv2.imwrite("matches.jpg", imMatches)
    # plt.imshow(imMatches)
    # plt.title("Keypoint matching of two images")
    # plt.axis('off')
    # plt.show()

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    M, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, M, (width, height))

    return im1Reg, M


# Main
if __name__ == '__main__':
    images = [cv2.imread(file) for file in glob.glob("*.jpg")]
    for i in range(len(images) - 1):
        print("|", i + 1, "images being aligned |")
        imgReg, m = alignImages(images[i + 1], images[0])
        ##These two line will blue the array of image
        # imgblur = cv2.blur(imgReg,(25,25))
        # imgAligned.append(imgblur)
        ## end
        ##comment out this line if above blur or noise is in used.
        imgAligned.append(imgReg)
        ## end
print("---Displaying end result int pop-out window---")
stacked = np.uint8(np.median(imgAligned, axis=0))
sharpened = cv2.filter2D(stacked, -1, kernel)
sharpened9 = cv2.filter2D(stacked, -1, kernel9)
## Save output image
# cv2.imwrite("Meanstacksharpen5.jpg",sharpened)
# cv2.imwrite("Meanstack_only.jpg",stacked)
# cv2.imwrite("Meanstacksharpen9.jpg",sharpened9)
# cv2.imwrite("median stacked sharpen 5 .jpg",sharpened)
cv2.imwrite("median stacked.jpg", stacked)
cv2.imwrite("median stacked sharpen 9.jpg", sharpened9)
cv2.destroyAllWindows()
## Show the result image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
plt.title("Stacked and Sharpened Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow * cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB)
plt.title("Median Stacked Image")
plt.axis('off')
plt.show()
