import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 211780267

# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------

" Write a function which takes an image and returns the optical flow by using the LK algorithm:  "
def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    # Convert images to grayscale if necessary.
    # "All functions should be able to accept both gray-scale and color images"
    if len(im1.shape) > 2:  # if the shape has more than 2 dimensions
        img1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = im1
    if len(im2.shape) > 2:  # if the shape has more than 2 dimensions
        img2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = im2

    # check if win_size (window size) is odd.
    if win_size % 2 == 0:
        return "win_size must be an odd number"
    # proper handling of the window boundaries and ensures that the analysis is centered around the
    # desired pixel with an equal number of pixels on each side. Use in floor division.
    half_win_size = win_size // 2

    # calculate image gradients
    # "In order to compute the optical flow, you will first need to compute the gradients Ix and Iy and then
    # over a window centered around each pixel we calculate"
    Ix, Iy = cv2.Sobel(img2_gray, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(img2_gray, cv2.CV_64F, 0, 1, ksize=3)
    # now calculates the difference in pixel intensity values between the two images,
    # the 'It' represents the temporal gradient(difference in pixel intensity)
    It = img2_gray.astype(np.float64) - img1_gray.astype(np.float64)

    original_points = []  # for store the original points (pixel coordinates)
    vec_per_point = []  # for store the corresponding optical flow vectors for each point

    # iterate over image blocks with the step size from the input
    for row in range(half_win_size, im1.shape[0] - half_win_size, step_size):
        for column in range(half_win_size, im1.shape[1] - half_win_size, step_size):
            # extract windowed patches
            #  helps us by selecting and isolating a specific region of interest in the image for further
            #  analysis and estimation of optical flow.
            window = img2_gray[row - half_win_size: row + half_win_size + 1,
                     column - half_win_size: column + half_win_size + 1]
            # based on the current row and column indices:
            Ix_windowed = Ix[row - half_win_size: row + half_win_size + 1,
                          column - half_win_size: column + half_win_size + 1]
            Iy_windowed = Iy[row - half_win_size: row + half_win_size + 1,
                          column - half_win_size: column + half_win_size + 1]
            It_windowed = It[row - half_win_size: row + half_win_size + 1,
                          column - half_win_size: column + half_win_size + 1]
            # flatten the windowed gradient images into 1D arrays
            A = np.column_stack((Ix_windowed.flatten(), Iy_windowed.flatten()))
            b = -np.expand_dims(It_windowed.flatten(), axis=1)  # subtract a dimension

            ATA = np.dot(A.T, A)  # A.T it is the transpose matrix of A, and we calculate the matrixes multiplication
            ATA_eig_vals = np.linalg.eigvals(
                ATA)  # numpy function that computes the eigenvalues of matrix. I love python!!

            # the next check helps us filter out points where the optical flow estimation might be not good
            if ATA_eig_vals[0] <= 1 or ATA_eig_vals[1] / ATA_eig_vals[0] >= 100:
                continue

            curr_vec = np.dot(np.linalg.inv(ATA), b) # calculate the optical flow vector by multiplying the inverse of 'ATA' matrix with 'b' matrix
            original_points.append([column, row])  # append the current pixel to original_points list
            vec_per_point.append(
                [curr_vec[0, 0], curr_vec[1, 0]])  # append the corresponding optical flow vector to the second list
    return original_points, vec_per_point


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    pass


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    pass


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    pass


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    pass


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    pass


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    pass


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pass


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    pass


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    pass

