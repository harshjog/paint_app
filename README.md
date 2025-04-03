# EMNIST Paint App with Handwriting Recognition

This application combines a simple paint interface with a trained CNN model that recognizes handwritten characters in real-time. Users can draw characters which are then classified using a model trained on the EMNIST Balanced dataset.

## Features

- Interactive drawing canvas with multiple color options
- Real-time handwritten character recognition
- Support for digits, uppercase and lowercase letters (47 classes)
- Simple user interface with keyboard controls

## Requirements

- TensorFlow 2.x
- NumPy
- OpenCV (cv2)
- Matplotlib
- Pre-trained model weights: "check_pt_emnist.weights.h5"

## Controls

- **Draw**: Left-click and drag on the canvas
- **Change Color**: Click on the color bar on the right side of the canvas
  - Blue (top)
  - Green
  - Red
  - Black
  - White/Eraser (bottom)
- **Change Brush Size**: Press 's' key, then enter an integer in the command window
- **Clear Canvas**: Press 'c' key
- **Recognize Character**: Press 'n' key after drawing
- **Quit**: Press 'q' key

## Color Selection

The right edge of the canvas contains five color options:
- Blue
- Green
- Red
- Black
- White

## Character Recognition

The application uses a CNN model trained on the EMNIST Balanced dataset to recognize:
- Digits (0-9)
- Uppercase letters (A-Z)
- Lowercase letters (a-z)

When you press 'n', the application:
1. Captures the current drawing
2. Resizes it to 28x28 pixels
3. Processes it for the neural network
4. Outputs the recognized character to the console

## Model Architecture

The CNN model consists of:
- 3 convolutional layers with increasing filters (64, 128, 256)
- Max pooling layers after each convolution
- Dropout regularization (50%)
- A final dense layer with 47 output classes

## Setup

1. Ensure you have the pre-trained model weights file "check_pt_emnist.weights.h5" in the same directory. 
1.1. If not, ensure you have the .zip file from the emnist database (https://www.nist.gov/itl/products-and-services/emnist-dataset) extracted in the same folder and run the recog.py script to generate it:
   ```
   python recog.py
   ```

2. Run the script:
   ```
   python paint.py
   ```
3. Draw characters on the canvas and press 'n' to recognize them, press 'c' to clear the canvas.

## Troubleshooting

- If recognition accuracy is low, try drawing larger, clearer characters
- The confidence threshold is set to 0.6 by default, which can be adjusted in the code
- Make sure your drawing stays within the main canvas area (not in the color selection area)
