import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import pickle
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import collections  # New import for prediction smoothing
from tensorflow.keras.regularizers import l2

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 1. CREATE DATASET FUNCTIONS
def collect_hand_landmarks_dataset(num_samples_per_digit=100):
    """Collect hand landmarks data for digits 0-9"""
    # Create directory for the dataset if it doesn't exist
    dataset_dir = 'hand_landmarks_dataset'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # Initialize data collection variables
    landmarks_data = []
    labels = []
    
    # Initialize the webcam
    cap = cv2.VideoCapture(1)  # Use 0 for default camera, or 1 for external webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7) as hands:
        
        # Collect data for each digit (0-9)
        for digit in range(10):
            print(f"\nShow digit {digit} with your hand.")
            print(f"Collecting {num_samples_per_digit} samples for digit {digit}...")
            print("Press 's' to start collecting, 'q' to quit")
            
            samples_collected = 0
            collecting = False
            
            while samples_collected < num_samples_per_digit:
                # Read frame from webcam
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break
                
                # Flip the frame horizontally for a more intuitive mirror view
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame to detect hands
                results = hands.process(rgb_frame)
                
                # Display instructions
                instructions = f"Digit: {digit} | Samples: {samples_collected}/{num_samples_per_digit}"
                cv2.putText(frame, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if collecting:
                    cv2.putText(frame, "COLLECTING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Press 's' to start collecting", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Draw hand landmarks if detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # If collecting and hand is detected, save landmarks
                        if collecting:
                            # Extract landmark coordinates (x, y, z) for all 21 hand landmarks
                            landmarks = []
                            for landmark in hand_landmarks.landmark:
                                landmarks.extend([landmark.x, landmark.y, landmark.z])
                            
                            # Add to dataset
                            landmarks_data.append(landmarks)
                            labels.append(digit)
                            samples_collected += 1
                            
                            # Show progress
                            print(f"\rCollected {samples_collected}/{num_samples_per_digit} samples for digit {digit}", end="")
                
                # Display the frame
                cv2.imshow('Hand Gesture Dataset Collection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    collecting = False
                    break
                elif key == ord('s'):
                    collecting = True
            
            print(f"\nFinished collecting samples for digit {digit}")
            
            # If user quit, break out of the digit loop
            if key == ord('q'):
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Convert to numpy arrays
    X = np.array(landmarks_data)
    y = np.array(labels)
    
    # Save the dataset
    with open(os.path.join(dataset_dir, 'hand_landmarks_dataset.pkl'), 'wb') as f:
        pickle.dump({'features': X, 'labels': y}, f)
    
    print("\nDataset collection completed!")
    print(f"Total samples collected: {len(X)}")
    return X, y

def load_dataset(filepath='hand_landmarks_dataset/hand_landmarks_dataset.pkl'):
    """Load the hand landmarks dataset"""
    if not os.path.exists(filepath):
        print("Dataset not found. Please collect data first.")
        return None, None
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data['features'], data['labels']

def collect_additional_samples(digits_to_improve=[5, 6, 7, 8, 9], samples_per_digit=50):
    """Collect additional samples for specific digits that need improvement"""
    # Load existing dataset
    dataset_path = 'hand_landmarks_dataset/hand_landmarks_dataset.pkl'
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    existing_features = data['features']
    existing_labels = data['labels']
    
    # Initialize data collection variables
    new_landmarks = []
    new_labels = []
    
    # Initialize the webcam
    cap = cv2.VideoCapture(1)  # Use 0 for default camera, or 1 for external webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7) as hands:
        
        # Collect data for each digit to improve
        for digit in digits_to_improve:
            print(f"\nShow digit {digit} with your hand.")
            print(f"Collecting {samples_per_digit} additional samples...")
            print("Try different angles and distances from the camera")
            print("Press 's' to start collecting, 'q' to quit")
            
            samples_collected = 0
            collecting = False
            
            while samples_collected < samples_per_digit:
                # Read frame from webcam
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame
                results = hands.process(rgb_frame)
                
                # Display instructions
                cv2.putText(frame, f"Digit: {digit} | Samples: {samples_collected}/{samples_per_digit}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if collecting:
                    cv2.putText(frame, "COLLECTING...", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Press 's' to start collecting", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # If collecting and hand is detected, save landmarks
                        if collecting:
                            landmarks = []
                            for landmark in hand_landmarks.landmark:
                                landmarks.extend([landmark.x, landmark.y, landmark.z])
                            
                            # Add to dataset
                            new_landmarks.append(landmarks)
                            new_labels.append(digit)
                            samples_collected += 1
                            
                            print(f"\rCollected {samples_collected}/{samples_per_digit} samples for digit {digit}", end="")
                
                # Display the frame
                cv2.imshow('Additional Data Collection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    collecting = False
                    break
                elif key == ord('s'):
                    collecting = True
            
            print(f"\nFinished collecting additional samples for digit {digit}")
            
            if key == ord('q'):
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Combine with existing dataset
    combined_features = np.vstack([existing_features, np.array(new_landmarks)])
    combined_labels = np.append(existing_labels, np.array(new_labels))
    
    # Save the updated dataset
    with open(dataset_path, 'wb') as f:
        pickle.dump({'features': combined_features, 'labels': combined_labels}, f)
    
    print("\nDataset updated successfully!")
    print(f"Added {len(new_landmarks)} new samples")
    print(f"Total samples now: {len(combined_features)}")
    
    return combined_features, combined_labels

def collect_hand_gesture_images(num_samples_per_class=100):
    # Create a folder for the dataset if needed
    dataset_dir = 'hand_gesture_images'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    cap = cv2.VideoCapture(1)
    all_images = []
    all_labels = []
    
    # Loop over each class (digit 0-9)
    for digit in range(10):
        print(f"Collecting images for digit {digit}. Press 's' to start and 'q' to quit.")
        samples_collected = 0
        while samples_collected < num_samples_per_class:
            ret, img = cap.read()
            if not ret:
                continue
            # Optional: flip/mirror depending on desired view
            img = cv2.flip(img, 1)
            # Preprocess the image (get the 64x64 grayscale image)
            img_final, _ = preprocess_image(img)
            # Remove the batch dimension -> shape becomes (64,64,1)
            img_final = img_final[0]
            
            cv2.putText(img, f"{digit}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Collect", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # start collecting
                all_images.append(img_final)
                all_labels.append(digit)
                samples_collected += 1
                print(f"Collected {samples_collected}/{num_samples_per_class} for digit {digit}")
            elif key == ord('q'):
                break
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return np.array(all_images), np.array(all_labels)

# 2. CREATE MODEL FOR HAND GESTURE RECOGNITION
def create_hand_gesture_model(input_dim, num_classes=10):
    """Create a neural network for hand gesture recognition based on landmarks"""
    
    inputs = Input(shape=(input_dim,))
    
    # First dense layer
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second dense layer
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_improved_hand_gesture_model(input_dim, num_classes=10):
    """Create an improved model with more capacity to distinguish similar gestures"""
    
    inputs = Input(shape=(input_dim,))
    
    # First dense layer - bigger
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second dense layer
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Third dense layer
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Compile with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_complex_hand_gesture_model(input_dim, num_classes=10):
    """Create a more complex and regularized neural network for hand gesture recognition with fine tuning"""
    inputs = Input(shape=(input_dim,))
    
    # First dense block
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  # Reduced dropout for fine tuning
    
    # Second dense block
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  # Reduced dropout
    
    # Third dense block
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Fourth dense block
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025),  # Lowered learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_cnn_hand_gesture_model(input_shape=(64, 64, 1), num_classes=10):
    """
    Create a CNN model for hand gesture recognition.
    Input: Grayscale image of size 64x64 (shape=(64,64,1)).
    Output: 10-class softmax predictions.
    """
    model = Sequential()
    # First convolutional block
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    # Second block
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    # Third block
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. TRAIN MODEL
def train_hand_gesture_model(X, y):
    """Train the hand gesture recognition model"""
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Split training data to create validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Create model
    input_dim = X_train.shape[1]  # Number of features (21 landmarks × 3 coordinates)
    model = create_hand_gesture_model(input_dim, num_classes=10)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint('best_hand_gesture_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,  # More epochs with early stopping
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('hand_gesture_training_history.png')
    plt.show()
    
    return model

def normalize_landmarks(landmarks):
    """Normalize hand landmarks to be invariant to scale and translation"""
    # Reshape landmarks array to separate x, y, z coordinates
    landmarks_array = np.array(landmarks).reshape(-1, 3)
    
    # Get wrist position (typically the first landmark)
    wrist = landmarks_array[0]
    
    # Center landmarks around wrist
    centered = landmarks_array - wrist
    
    # Find the scale (maximum distance from any landmark to wrist)
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale > 0:
        # Normalize by scale
        normalized = centered / scale
    else:
        normalized = centered
    
    # Flatten back to original shape
    return normalized.flatten()

def augment_landmarks(landmarks, noise_std=0.01, num_augments=3):
    """
    Generate augmented copies of a landmark sample by adding Gaussian noise.
    noise_std: standard deviation of the noise (adjust based on normalized scale)
    num_augments: number of augmented copies per original sample
    """
    augmented = []
    landmarks_array = np.array(landmarks)
    for _ in range(num_augments):
        noise = np.random.normal(0, noise_std, landmarks_array.shape)
        augmented.append(landmarks_array + noise)
    return augmented

def train_with_class_weighting(X, y):
    """Train the model using a more complex model with class weighting and data augmentation."""
    # Normalize each training sample using the same normalize_landmarks function
    X_norm = np.array([normalize_landmarks(sample) for sample in X])
    
    # Augment the dataset – for every sample, add a few jittered copies
    X_aug_list = []
    y_aug_list = []
    for i, sample in enumerate(X_norm):
        aug_samples = augment_landmarks(sample, noise_std=0.01, num_augments=3)
        X_aug_list.extend(aug_samples)
        y_aug_list.extend([y[i]] * len(aug_samples))
    
    print(f"Original samples: {len(X_norm)}")
    print(f"Augmented samples added: {len(X_aug_list)}")
    
    # Combine original and augmented data
    X_total = np.concatenate((X_norm, np.array(X_aug_list)), axis=0)
    y_total = np.concatenate((y, np.array(y_aug_list)), axis=0)
    
    # Split data into train, validation and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # One-hot encode labels
    y_train_cat = to_categorical(y_train, 10)
    y_val_cat = to_categorical(y_val, 10)
    y_test_cat = to_categorical(y_test, 10)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Define class weights (to focus on problematic digits)
    class_weights = {
        0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0,
        5: 2.0, 6: 2.0, 7: 2.0, 8: 2.0, 9: 2.0
    }
    
    input_dim = X_train.shape[1]
    # Use your fine-tuned complex model
    model = create_complex_hand_gesture_model(input_dim, num_classes=10)
    
    # Use callbacks adjusted for fine tuning
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint('best_hand_gesture_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=0.000001, verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train_cat,
        epochs=500,
        batch_size=32,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Calculate accuracy using sklearn's function
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_test_cat, axis=1)
    acc = accuracy_score(true_classes, y_pred_classes)
    print("Sklearn Accuracy Score:", acc)
    
    return model

# 4. REAL-TIME RECOGNITION WITH MEDIAPIPE
def real_time_hand_gesture_recognition(model=None):
    """Real-time hand gesture recognition with prediction smoothing using MediaPipe"""
    if model is None:
        if os.path.exists('best_hand_gesture_model.h5'):
            model = load_model('4best_hand_gesture_model.h5')
        else:
            print("Model not found. Please train the model first.")
            return

    print("Starting real-time recognition. Press 'q' to exit.")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Buffer for smoothing predictions (last 5 predictions)
    prediction_buffer = collections.deque(maxlen=5)

    # Initialize MediaPipe Hands (already declared at top)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Flip for a mirror view (optional)
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks and results.multi_handedness:
                # Process each detected hand along with its handedness info.
                for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Extract landmarks from MediaPipe (already normalized to [0,1])
                    landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark]
                    
                    # Mirror right-hand landmarks: MediaPipe returns landmarks in normalized coordinates.
                    # When the hand is "Right", we mirror by replacing x with (1 - x).
                    hand_label = hand_info.classification[0].label   # "Left" or "Right"
                    if hand_label == "Right":
                        landmarks = [[1 - pt[0], pt[1], pt[2]] for pt in landmarks]
                    
                    # Flatten and then normalize landmarks so they are translation and scale invariant.
                    # (normalize_landmarks expects a flat array or an array reshaped to (-1,3))
                    landmarks_flat = np.array(landmarks).flatten()
                    landmarks_normalized = normalize_landmarks(landmarks_flat)
                    
                    # Predict the gesture using the model
                    prediction = model.predict(np.array([landmarks_normalized]), verbose=0)
                    index = np.argmax(prediction[0])
                    confidence = prediction[0][index]
                    
                    # Append to prediction buffer for smoothing
                    prediction_buffer.append(index)
                    smoothed_pred = max(set(prediction_buffer), key=prediction_buffer.count)
                    
                    # Draw prediction on the frame
                    cv2.putText(frame, f"{smoothed_pred} ({confidence:.2f})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # Optionally, you can also draw the landmarks using mp_drawing.
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            cv2.imshow("Real-Time Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

def evaluate_with_confusion_matrix(model, X, y):
    """Evaluate model and show confusion matrix to identify problem digits"""
    # Get predictions
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred_classes))

# 5. MAIN EXECUTION
if __name__ == "__main__":
    # Check if dataset exists
    dataset_path = 'hand_landmarks_dataset/hand_landmarks_dataset.pkl'
    
    if not os.path.exists(dataset_path):
        print("Dataset not found. Starting data collection...")
        X, y = collect_hand_landmarks_dataset(num_samples_per_digit=10)
    else:
        # Ask if user wants to collect new data or use existing
        choice = input("Dataset found. Do you want to [t]rain on existing data, [c]ollect new data, or [r]un recognition? (t/c/r): ").lower()
        
        if choice == 'c':
            X, y = collect_hand_gesture_images(num_samples_per_class=10)
        elif choice == 't':
            X, y = load_dataset(dataset_path)
            if X is not None and y is not None:
                model = train_with_class_weighting(X, y)
        elif choice == 'r':
            real_time_hand_gesture_recognition()
        else:
            print("Invalid choice. Exiting.")

    # -------------------------------
    # Updated Evaluation Code:
    # -------------------------------
    
    # Load dataset previously collected (adjust the file path as needed)
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    # X_raw contains raw landmark features as collected (length 63 per sample)
    X_raw = np.array(data['features'])
    # IMPORTANT: Normalize each sample using the same function as used during training
    X = np.array([normalize_landmarks(sample) for sample in X_raw])
    y = np.array(data['labels'])      # Labels remain as integers (0-9)

    # Split data into training and testing sets (80-20 split)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert labels to one-hot vectors for evaluation if using categorical_crossentropy
    from tensorflow.keras.utils import to_categorical
    y_test_cat = to_categorical(y_test, num_classes=10)

    # Evaluate the model using the evaluate function (returns loss and accuracy)
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print("Model Evaluate Test Accuracy:", accuracy)

    # Alternatively, calculate accuracy manually:
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, predicted_classes)
    print("Sklearn Accuracy Score:", acc)

    # Example training code:
    X, y = collect_hand_gesture_images(num_samples_per_class=50)  # Adjust sample count as needed
    X = X.astype('float32')  # Already normalized by preprocess_image (divided by 255)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes=10)
    y_test_cat = to_categorical(y_test, num_classes=10)
    
    # Create CNN model
    model = create_cnn_hand_gesture_model(input_shape=(64,64,1), num_classes=10)
    
    # Train the CNN
    history = model.fit(
        X_train, y_train_cat,
        epochs=50,  # adjust epochs as needed
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print("CNN Test Accuracy:", test_acc) 