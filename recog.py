import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import gzip
import shutil
import emnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
import scipy
import cv2
from pathlib import Path

# Function to fix image orientation because emnist data seems to be oriented differently
def fix_image_orientation(image):
    # Rotate 90 degrees clockwise
    rotated = np.rot90(image, k=-1)
    # Flip horizontally (mirror flip)
    fixed_image = np.fliplr(rotated)
    return fixed_image


dataset_path = "./emnist/"
files = {
    "train_images": "emnist-balanced-train-images-idx3-ubyte.gz",
    "train_labels": "emnist-balanced-train-labels-idx1-ubyte.gz",
    "test_images": "emnist-balanced-test-images-idx3-ubyte.gz",
    "test_labels": "emnist-balanced-test-labels-idx1-ubyte.gz"
}

def load_emnist_images(filename):
    with gzip.open(os.path.join(dataset_path, filename), 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28, 1)

def load_emnist_labels(filename):
    with gzip.open(os.path.join(dataset_path, filename), 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)

# Load the dataset
X_train = load_emnist_images(files["train_images"])
y_train = load_emnist_labels(files["train_labels"])
X_test = load_emnist_images(files["test_images"])
y_test = load_emnist_labels(files["test_labels"])

# Apply the fix to the entire dataset
X_train = np.array([fix_image_orientation(img) for img in X_train])
X_test = np.array([fix_image_orientation(img) for img in X_test])

print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
print(f"Testing set: {X_test.shape}, Labels: {y_test.shape}")

# convert image data into floating point numbers with values from 0-1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# invert black and white parts of image
X_train = 1 - X_train
X_test = 1 - X_test

# for debugging
#plt.imshow(X_train[0].squeeze(), cmap="gray")
#plt.title(f"Label: {y_train[0]}")
#plt.show()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

num_classes = len(np.unique(y_train))
print("Number of unique classes:", num_classes)  # Should print 47

model_cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), padding = "SAME", activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3,3), padding = "SAME", activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Helps prevent overfitting
    tf.keras.layers.Dense(47, activation='softmax')  # 47 classes
])


model_cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Saving the model parameters into a .h5 file to use in other apps
checkpoint_path = "check_pt_emnist_re.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Data augmentation for more robust detection
datagen = ImageDataGenerator(
    rotation_range=10, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    zoom_range=0.1)

model_cnn.fit(datagen.flow(X_train, y_train, batch_size=64),
              epochs=25,
              validation_data=(X_test, y_test),
              callbacks=[cp_callback])

# Save neural network structure
model_structure = model_cnn.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

probability_model = tf.keras.Sequential([model_cnn,
                                        tf.keras.layers.Softmax()])

predictions = probability_model.predict(X_test)