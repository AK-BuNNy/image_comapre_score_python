import cv2
import numpy as np  # type: ignore

def compare_sketches(ref_path, cand_path):
    # Read images
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    cand_img = cv2.imread(cand_path, cv2.IMREAD_GRAYSCALE)

    # Preprocess images (thresholding and resizing)
    ref_img = cv2.resize(ref_img, (500, 500))
    cand_img = cv2.resize(cand_img, (500, 500))
    _, ref_thresh = cv2.threshold(ref_img, 127, 255, cv2.THRESH_BINARY)
    _, cand_thresh = cv2.threshold(cand_img, 127, 255, cv2.THRESH_BINARY)

    # Contour detection
    ref_contours, _ = cv2.findContours(ref_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand_contours, _ = cv2.findContours(cand_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compare contours using Hu Moments
    ref_hu_moments = cv2.HuMoments(cv2.moments(ref_contours[0])).flatten()
    cand_hu_moments = cv2.HuMoments(cv2.moments(cand_contours[0])).flatten()

    # Normalize moments and calculate similarity
    hu_similarity = sum(1 - abs(r - c) / (abs(r) + abs(c) + 1e-6) for r, c in zip(ref_hu_moments, cand_hu_moments))
    hu_score = (hu_similarity / len(ref_hu_moments)) * 100

    # Keypoint matching with ORB
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(ref_img, None)
    kp2, des2 = orb.detectAndCompute(cand_img, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Normalize match score
    keypoint_score = (len(matches) / max(len(kp1), len(kp2))) * 100

    # Weighted score
    final_score = 0.6 * hu_score + 0.4 * keypoint_score
    return final_score
