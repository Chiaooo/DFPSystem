import numpy as np
from PIL import Image
import tensorflow as tf


cropped_eye = Image.open('data/train/new_closed_eyes/eye_crop_closed0.jpg')
saved_model = tf.keras.models.load_model('output/best_model_5.h5')


def run_saved_model(model, eye_img):
    arr = np.asarray(eye_img)
    arr = arr.reshape(1, 80, 80, 3)
    prediction = model.predict(arr)[0][0].round()
    return prediction


prediction = run_saved_model(saved_model, cropped_eye)
print(prediction)
