from keras.models import load_model
from keras.preprocessing import image
import numpy as np
model = load_model('bird-model.h5')

img_name = './data/train/not_bird/Indigo_Bunting_0001_12469.jpg'

# Load a single image into x
width, height = 64, 64
img = image.load_img(img_name, target_size=(width, height))
img *= 255.0/np.array(img).max()
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# create an array with a single image
images = np.vstack([x])

# Test the class that the model is predicting
classes = model.predict_classes(images, batch_size=10)
print(classes)