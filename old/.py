import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
import collections

# Load the trained model
model = load_model('../best_hand_gesture_model.h5')

# Initialize video capture and hand detector
cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)

# Constants
offset = 20
imgSize = 64  # (Used for image preprocessing if needed; not used for landmark prediction)
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Create a prediction buffer for smoothing predictions
prediction_buffer = collections.deque(maxlen=5)

def normalize_landmarks(landmarks):
    """
    Normalize landmarks so that they are translation- and scale-invariant.
    Expects landmarks as either a list of 21 (x, y, z) lists or a flat array of length 63.
    """
    landmarks_array = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks_array[0]
    centered = landmarks_array - wrist
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered
    return normalized.flatten()

def preprocess_image(img):
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to match training size
    img = cv2.resize(img, (imgSize, imgSize))
    # Normalize
    img = img / 255.0
    # Reshape for model input
    img_final = np.expand_dims(img, axis=[0, -1])
    return img_final, img  # Return both the model input and the normalized image for display

while True:
    success, img = cap.read()
    if not success:
        break
        
    imgOutput = img.copy()
    img_h, img_w, _ = img.shape
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Extract landmarks (each landmark is [x, y, z] in pixel coordinates)
        landmarks = hand["lmList"]
        if landmarks is not None and len(landmarks) == 21:
            # Convert landmarks from pixel coordinates to normalized coordinates (0 to 1)
            landmarks_norm = []
            for pt in landmarks:
                # Divide x by image width and y by image height.
                x_norm = pt[0] / img_w
                y_norm = pt[1] / img_h
                landmarks_norm.append([x_norm, y_norm, pt[2]])
            
            # Mirror right-hand landmarks if hand type info is available.
            if "type" in hand and hand["type"] == "Right":
                landmarks_norm = [[1 - pt[0], pt[1], pt[2]] for pt in landmarks_norm]
            
            # Normalize landmarks (translation and scale invariant) to produce a 63-length vector
            landmarks_normalized = normalize_landmarks(landmarks_norm)
            
            # Predict using the model; expected input shape: (1, 63)
            prediction = model.predict(np.array([landmarks_normalized]), verbose=0)
            index = np.argmax(prediction[0])
            confidence = prediction[0][index]
            
            # Append prediction to the smoothing buffer
            prediction_buffer.append(index)
            smoothed_pred = max(set(prediction_buffer), key=prediction_buffer.count)
            
            # Draw prediction overlay on the output image
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, f"{labels[smoothed_pred]} ({confidence:.2f})", 
                        (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)
    
    cv2.imshow("Image", imgOutput)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 