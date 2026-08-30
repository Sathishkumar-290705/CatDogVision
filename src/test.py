import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

IMAGE_SIZE = (150,150)

model = tf.keras.models.load_model("dog_cat_model.h5")


def test(image_path):
    img = image.load_img(image_path,target_size = IMAGE_SIZE)
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)  
    prediction = model.predict(img_arr)[0][0]                
    print("Prediction accuracy ",prediction)
    if prediction > 0.5:
        print(f"Dog  (confidence: {prediction:.2f})")
    else:
        print(f"Cat  (confidence: {1 - prediction:.2f})")



if   __name__ == "__main__"  :
    IMG_PATH = r"E:\PROJECTS\CatDog Classifier\Data\test\cat_5.jpg"
    test(IMG_PATH)