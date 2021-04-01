import cv2

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from time import sleep

cam = cv2.VideoCapture("http://0.0.0.0:4747/mjpegfeed?640x480")

cv2.namedWindow("test")

model = load_model("./rps-cnn.h5")

names = ["paper", "rock", "scissor"]

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        pass
    img_name = "bruh.png"
    cv2.imwrite(img_name, frame)
    image = tf.keras.preprocessing.image.load_img("bruh.png", target_size=(300, 200))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    print(names[np.argmax(predictions)])
    #  sleep(2)


cam.release()

cv2.destroyAllWindows()
