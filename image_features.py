import cv2
import numpy as np
from math import sqrt
from scipy.spatial.distance import pdist
from skimage.feature import blob_dog, graycomatrix, graycoprops, local_binary_pattern

def extract_advanced_stats(img_gray, blobs):
    """
    Extracts morphological and spatial statistics from detected nuclei blobs.

    Args:
        img_gray (numpy.ndarray): The 2D grayscale image patch.
        blobs (numpy.ndarray): Array of detected blobs, where each row is (y, x, sigma).

    Returns:
        numpy.ndarray: A 1D array of 11 features including count, radius stats,
                       spatial distance stats, and pixel intensity stats.
    """
    if len(blobs) == 0:
        return np.zeros(11)

    radii = blobs[:, 2] * sqrt(2)
    count = len(blobs)
    mean_radius = np.mean(radii)
    std_radius = np.std(radii)

    coordinates = blobs[:, 0:2]
    if count > 1:
        distances = pdist(coordinates, metric='euclidean')
        mean_dist, min_dist, std_dist = np.mean(distances), np.min(distances), np.std(distances)
    else:
        mean_dist, min_dist, std_dist = 0.0, 0.0, 0.0

    intensities = []
    for y, x, r in blobs:
        y_int, x_int = int(y), int(x)
        if 0 <= y_int < img_gray.shape[0] and 0 <= x_int < img_gray.shape[1]:
            intensities.append(img_gray[y_int, x_int])

    if len(intensities) > 0:
        mean_intensity, std_intensity = np.mean(intensities), np.std(intensities)
        min_intensity, max_intensity = np.min(intensities), np.max(intensities)
        med_intensity = np.median(intensities)
    else:
        mean_intensity, std_intensity, min_intensity, max_intensity, med_intensity = 0.0, 0.0, 0.0, 0.0, 0.0

    return np.array([count, mean_radius, std_radius, mean_dist, min_dist, std_dist,
                     mean_intensity, std_intensity, min_intensity, max_intensity, med_intensity])

def extract_color_stats(img_bgr):
    """
    Extracts statistical color features by converting the image to HSV space.

    Args:
        img_bgr (numpy.ndarray): The 3D BGR image patch.

    Returns:
        numpy.ndarray: A 1D array of 6 features containing the mean and standard
                       deviation of the Hue, Saturation, and Value channels.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_mean, h_std = np.mean(img_hsv[:, :, 0]), np.std(img_hsv[:, :, 0])
    s_mean, s_std = np.mean(img_hsv[:, :, 1]), np.std(img_hsv[:, :, 1])
    v_mean, v_std = np.mean(img_hsv[:, :, 2]), np.std(img_hsv[:, :, 2])
    return np.array([h_mean, h_std, s_mean, s_std, v_mean, v_std])

def extract_glcm_features(img_gray):
    """
    Computes macro-texture properties using a Gray-Level Co-occurrence Matrix (GLCM).

    Args:
        img_gray (numpy.ndarray): The 2D grayscale image patch.

    Returns:
        numpy.ndarray: A 1D array of 5 texture features: contrast, dissimilarity,
                       homogeneity, energy, and correlation.
    """
    glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation])

def extract_lbp_features(img_gray):
    """
    Extracts micro-texture features using Local Binary Patterns (LBP).

    Args:
        img_gray (numpy.ndarray): The 2D grayscale image patch.

    Returns:
        numpy.ndarray: A 1D array representing the normalized histogram of the
                       uniform LBP image.
    """
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_lbglcm_features(img_gray):
    """
    Computes a hybrid feature set by applying GLCM to an LBP-transformed image.

    Args:
        img_gray (numpy.ndarray): The 2D grayscale image patch.

    Returns:
        numpy.ndarray: A 1D array of 5 GLCM texture properties extracted from
                       the LBP matrix.
    """
    radius = 1
    n_points = 8 * radius
    lbp_img = local_binary_pattern(img_gray, n_points, radius, method='uniform').astype(np.uint8)
    glcm = graycomatrix(lbp_img, distances=[1], angles=[0], levels=int(lbp_img.max() + 1), symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation])

def extract_glrlm_features(img_gray):
    """
    Extracts directional texture patterns using a Gray Level Run-Length Matrix (GLRLM).

    Args:
        img_gray (numpy.ndarray): The 2D grayscale image patch.

    Returns:
        numpy.ndarray: A 1D array of 5 run-length statistics: Short Run Emphasis (SRE),
                       Long Run Emphasis (LRE), Gray Level Non-uniformity (GLN),
                       Run Length Non-uniformity (RLN), and Run Percentage (RP).
    """
    img = (img_gray / 16).astype(np.uint8)
    rows, cols = img.shape
    run_lengths, total_runs = [], 0

    for row in img:
        run_val, run_len = row[0], 1
        for val in row[1:]:
            if val == run_val:
                run_len += 1
            else:
                run_lengths.append(run_len)
                total_runs += 1
                run_val, run_len = val, 1
        run_lengths.append(run_len)
        total_runs += 1

    run_lengths = np.array(run_lengths)
    if total_runs > 0:
        SRE = np.sum(1 / (run_lengths ** 2)) / total_runs
        LRE = np.sum(run_lengths ** 2) / total_runs
        RP = total_runs / (rows * cols)
        GLN = np.var(img.flatten())
        RLN = np.var(run_lengths)
    else:
        SRE = LRE = RP = GLN = RLN = 0.0
    return np.array([SRE, LRE, GLN, RLN, RP])

def extract_sfta_features(img_gray, n_levels=3):
    """
    Extracts fractal texture information using Segmentation-based Fractal Texture Analysis (SFTA).

    Args:
        img_gray (numpy.ndarray): The 2D grayscale image patch.
        n_levels (int, optional): The number of threshold levels to compute. Defaults to 3.

    Returns:
        numpy.ndarray: A 1D array containing the area, mean, and standard deviation
                       for each of the binary regions generated by the thresholds.
    """
    thresholds = np.linspace(img_gray.min(), img_gray.max(), n_levels + 2)[1:-1]
    features = []
    for t in thresholds:
        binary = (img_gray > t).astype(np.uint8)
        area = binary.sum()
        if area > 0:
            mean_val = img_gray[binary == 1].mean()
            std_val = img_gray[binary == 1].std()
        else:
            mean_val = std_val = 0.0
        features.extend([area, mean_val, std_val])
    return np.array(features)