import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import math
import os
import string

# Global variables
canvas_height, canvas_width = 800, 800
canvas = np.ones([canvas_height,canvas_width,3],'uint8')*255
canvas[-50:, -50:] = (255,255,255)
canvas[-100:-50, -50:] = (255, 0, 0)
canvas[-150:-100, -50:] = (0, 255, 0)
canvas[-200:-150, -50:] = (0, 0, 255)
canvas[-250:-200, -50:] = (0, 0, 0)
startpt = (0,0)
endpt = (0,0)
color = (255,0,0)
line_width = -1      #-1 means circle is filled
radius = 20
lastevent = 1
threshold = 0.6

print("default color is blue")

# click callback
def click(event, x, y, flags, param):
    global canvas, color, lastevent #,startpt, movept, endpt,
    
    if event == cv2.EVENT_LBUTTONDOWN:
        #print("LButton Down")
        #print("Start",x,y)
        startpt = (x,y)
        cv2.circle(canvas, startpt, radius, color, line_width)
        lastevent = 0
    elif event == cv2.EVENT_MOUSEMOVE:
        #print("Mouse Move")
        movept = (x,y)
        if lastevent == 0:
            cv2.circle(canvas, movept, radius, color, line_width)
    elif event == cv2.EVENT_LBUTTONUP:
        #print("LButton Up")
        #print("End",x,y)
        lastevent = 1
        endpt = (x,y)
        if endpt[0] >= (canvas_width-50) and endpt[1] >= (canvas_height-50):
            color = (255, 255, 255)
            print("Eraser mode")
        elif endpt[0] >= (canvas_width-50) and endpt[1] > (canvas_height-100):
            color = (255, 0, 0)
            print("color changed to blue")
        elif endpt[0] >= (canvas_width-50) and endpt[1] > (canvas_height-150):
            color = (0, 255, 0)
            print("color changed to green")
        elif endpt[0] >=(canvas_width-50) and endpt[1]> (canvas_height-200):
            color = (0, 0, 255)
            print("color changed to red")
        elif endpt[0] >=(canvas_width-50) and endpt[1]> (canvas_height-250):
            color = (0,0,0)
            print("color changed to black")
        cv2.circle(canvas, endpt, radius, color, line_width)

# window initialization and callback assignment
cv2.namedWindow("canvas")
cv2.setMouseCallback("canvas", click)

############ DNN module for digit recognition

checkpoint_path = "check_pt_emnist.weights.h5"

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

model_cnn.load_weights(checkpoint_path)

probability_model = tf.keras.Sequential([model_cnn,
                                        tf.keras.layers.ReLU()])
# EMNIST Balanced Mapping: Digits + Uppercase + Some Lowercase
emnist_labels = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)  # 47 characters
#print(emnist_labels)

def get_character(label):
    return emnist_labels[label]

# Forever draw loop
while True:
    
    cv2.imshow("canvas",canvas)
    
	# key capture every 1ms
    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('c'):
        canvas = np.ones([canvas_height,canvas_width,3],'uint8')*255
        canvas[-50:, -50:] = (255,255,255)
        canvas[-100:-50, -50:] = (255, 0, 0)
        canvas[-150:-100, -50:] = (0, 255, 0)
        canvas[-200:-150, -50:] = (0, 0, 255)
        canvas[-250:-200, -50:] = (0, 0, 0)
    if ch & 0xFF == ord('s'):
        while True:
            try:
                radius = int(input("Enter the brush size: "))
                break  # Exit loop if input is valid
            except ValueError:
                print("Invalid input. Please enter a number.")
        print("You entered:", radius)
    if ch & 0xFF == ord('n'):
        img_resize = cv2.resize(canvas[:,:-50], (28,28))
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite("test.jpg", img_resize)
        img_resize = np.expand_dims(img_resize, axis=0)  # Add batch dimension
        img_resize = np.expand_dims(img_resize, axis=-1)  # Add channel dimension
        prediction = probability_model.predict(img_resize)
        index = np.where(prediction[:] > threshold)
        #print(prediction)
        label_number = index[1][0]
        print(get_character(label_number))
        #print(prediction.shape)
        #print(index[1][0])
        
    if ch & 0xFF == ord('q'):
        break
	

cv2.destroyAllWindows()