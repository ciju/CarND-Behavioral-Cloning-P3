# Try with Tensorflow 1.1.0 and keras 2.0.3
# Example: https://github.com/udacity/CarND-Term1-Starter-Kit/pull/73/files
# Trained on track 1 sample data 

import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D

lines = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

lines = lines[1:]

dataSize = len(lines)
batchSize = 30

def brightness_shift(img, bright_value=None):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if bright_value:
        img[:,:,2] += bright_value
    else:
        random_bright = .25 + np.random.uniform()
        img[:,:,2] = img[:,:,2] * random_bright

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

# Effected the precisions a lot
# https://discussions.udacity.com/t/convert-color-of-training-images-bgr-to-rgb/372400
def readImg(imgPath):
    return brightness_shift(cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB))

def xyForImg(line):
    steering_center = float(line[3])

    correction = 0.20
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    current_path = 'data/' + line[0].strip()
    left_path = 'data/' + line[1].strip()
    right_path = 'data/' + line[2].strip()

    imgs = [readImg(i) for i in [current_path, left_path, right_path]]
    steerings = [steering_center, steering_left, steering_right]

    return imgs, steerings

def myGenerator(lines):
    while True:
        # Shuffle data
        lines = sklearn.utils.shuffle(lines)
        for i in range(0, dataSize, batchSize):
            batchImgs = []
            batchSteerings = []
            for line in lines[i: i + batchSize]:
                imgs, steerings = xyForImg(line)
                batchImgs.extend(imgs)
                batchSteerings.extend(steerings)

            # If the batch has no images
            if (len(batchImgs) < 1):
                continue

            batchImgs.extend([cv2.flip(i, 1) for i in batchImgs])
            batchSteerings.extend([(i * -1.0) for i in batchSteerings])

            yield sklearn.utils.shuffle(np.array(batchImgs), np.array(batchSteerings))


model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((75, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Dropout(.1))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Dropout(.1))
model.add(Conv2D(48, (3, 3), strides=(2, 2), activation='relu'))
model.add(Dropout(.1))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Dense(54))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

train_lines, val_lines = train_test_split(lines, test_size=0.2)
model.fit_generator(
    myGenerator(train_lines),
    validation_data=myGenerator(val_lines),
    validation_steps=int(len(val_lines)/batchSize),
    steps_per_epoch=int(len(train_lines)/batchSize),
    epochs=3,
    verbose=2
)
print('saving model')
model.save('model.h5')
exit()
