import os
import cv2
import yaml
import numpy as np
from mtcnn import MTCNN
from utils import get_subdirectories_paths, get_file_paths, load_image, plot_image, save_image  

def align_face(image, landmarks, H, W):
    """
    Aligns a face in an image based on detected facial landmarks.

    Parameters:
    image (numpy.ndarray): The input image containing the face to be aligned.
    landmarks (dict): A dictionary containing the detected facial landmarks with keys:
                      'left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right'.
    H (int): The desired height of the output aligned image.
    W (int): The desired width of the output aligned image.

    Returns:
    numpy.ndarray: The aligned face image with dimensions (H, W).
    """

    # Standard facial landmarks (center of the eyes, tip of the nose, corners of the mouth)
    desired_landmarks = np.array([
        [0.381 * W, 0.517 * H],  # Left eye
        [0.735 * W, 0.515 * H],  # Right eye
        [0.560 * W, 0.717 * H],  # Tip of the nose
        [0.415 * W, 0.924 * H],  # Left mouth corner
        [0.707 * W, 0.922 * H]   # Right mouth corner
    ], dtype=np.float32)

    # Detected landmarks from MTCNN
    detected_landmarks = np.float32([
        landmarks['left_eye'],
        landmarks['right_eye'],
        landmarks['nose'],
        landmarks['mouth_left'],
        landmarks['mouth_right']
    ])
    
    # Calculate the affine transformation matrix
    transformation_matrix, _ = cv2.estimateAffinePartial2D(detected_landmarks, desired_landmarks, method=cv2.LMEDS)
    
    # Apply the transformation to align the face
    aligned_image = cv2.warpAffine(image, transformation_matrix, (W, H))
    
    return aligned_image

def preprocess_face():
    """
    Preprocess faces in a dataset by detecting and aligning them.
    Returns:
    None
    """
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    dataset = config['RAW_ROOT']
    prs_face = config['PROCESSED_ROOT']
    not_prs_face = config['NOT_PROCESSED_ROOT']
    H = config['H']
    W = config['W']
    detector = MTCNN()
    subs = get_subdirectories_paths(dataset)

    for sub in subs:
        paths = get_file_paths(sub)
        for path in paths:
            img = load_image(path)
            faces = detector.detect_faces(img)
            if len(faces) == 0:
                save_image(img, os.path.join(not_prs_face, os.path.basename(sub)[5:], os.path.basename(path)))
            else:
                for face in faces:
                    box = face['box']
                    landmarks = face['keypoints']

                    # Crop face from the original image
                    x, y, width, height = box
                    face_image = img[y:y+height, x:x+width]

                    # Adjust the coordinates of the landmarks according to the cropped region
                    adjusted_landmarks = {
                        'left_eye': (landmarks['left_eye'][0] - x, landmarks['left_eye'][1] - y),
                        'right_eye': (landmarks['right_eye'][0] - x, landmarks['right_eye'][1] - y),
                        'nose': (landmarks['nose'][0] - x, landmarks['nose'][1] - y),
                        'mouth_left': (landmarks['mouth_left'][0] - x, landmarks['mouth_left'][1] - y),
                        'mouth_right': (landmarks['mouth_right'][0] - x, landmarks['mouth_right'][1] - y)
                    }

                    # Align the face
                    aligned_face = align_face(face_image, adjusted_landmarks, H, W)
                    save_image(aligned_face, os.path.join(prs_face, os.path.basename(sub)[5:], os.path.basename(path)))