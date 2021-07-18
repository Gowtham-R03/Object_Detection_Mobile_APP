import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
from glob import glob
import pathlib
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.np_utils import to_categorical  # for one hot encoding

# re-size all the images to this
IMAGE_SIZE = [224, 224]

# train_path = 'Datasets/Train'
# valid_path = 'Datasets/Test'

# add preprocessing layer to the front of VGG
inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in inception.layers:
    layer.trainable = False

# useful for getting number of classes
folders = glob('datasets/flower_photos/*')

# Flower classification #########################
data_dir = 'datasets/flower_photos'

data_dir = pathlib.Path(data_dir)  # converting to windows path

image_count = len((list(data_dir.glob('*/*.jpg'))))  # list all the datasets with jpg extension
print(image_count)

# to see images of specific classes

roses = list(data_dir.glob('roses/*'))
print(roses[:10])

# to show images using PIL

# Image.open(str(roses[2])).show()

dandelion = list(data_dir.glob('dandelion/*'))
# Image.open(str(dandelion[10])).show()

# create class names
# image dictionary
flowers_images_dict = {
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'roses': list(data_dir.glob('roses/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'tulips': list(data_dir.glob('tulips/*')),
}
# labels dictionary
flowers_labels_dict = {
    'daisy': 0,
    'dandelion': 1,
    'roses': 2,
    'sunflowers': 3,
    'tulips': 4,
}

# reading images
img = cv2.imread(str(flowers_images_dict['roses'][3]))
# cv2.imshow('Image', img)
# cv2.waitKey(0)
print(img.shape)

# the images shape are not unique but we need all images size to be unique
img1 = cv2.resize(img, IMAGE_SIZE)
print(img1.shape)

X, y = [], []

for flower_name, images in flowers_images_dict.items():
    # print(flower_name)
    # print(len(images))
    for image in images:
        img = cv2.imread(str(image))
        img = cv2.resize(img, IMAGE_SIZE)
        X.append(img)
        y.append(flowers_labels_dict[flower_name])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)  # for validation

# scalling
X_train = X_train / 255
X_test = X_test / 255
noOfClasses = len(folders)
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

# our layers - you can add more if you want
x = Flatten()(inception.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=inception.input, outputs=prediction)

# view the structure of the model
print(model.summary())

# tell the model what cost and optimization method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

################## Augmentation (To make our image generic)

dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,
                             rotation_range=10)

dataGen.fit(X_train)


history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=32), steps_per_epoch=int(len(X_train)/32), epochs=10, validation_data=(X_validation, y_validation), shuffle=1)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])


print(y_test[10])

preds = model.predict(X_test)
print(np.argmax(preds[10]))

############################### PLOT
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()

# save model
model.save('my_new_model.h5')
