from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import SGD


train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

width, height = 64, 64
train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(width, height),
        batch_size=32)

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(width, height),
        batch_size=32)        

model = Sequential()
model.add(Conv2D(16, (5, 5), input_shape=(width, height, 3)))
model.add(Flatten())

model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=2, activation="softmax"))


model.compile(optimizer='adam',loss="mean_squared_error",metrics=["accuracy"])

model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=4,
        validation_data=validation_generator,
        validation_steps=100)
model.summary()

model.save('./bird-model.h5')