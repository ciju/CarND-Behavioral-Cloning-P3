# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* Readme.md summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I mostly followed the approach followed in the course videos. Picked up nVidia model from there. Somehow initially I ignored subsampling for convolution layers. Which lead to a lot of time wasted in trying out other parameter changes. But most of the time, the network was over-fitting. And wasn't generalized enough in real runs. Realized the mistake, and added subsampling and dropouts to accommodate for over-fitting.

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 85-92) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (model.py line 83). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers and subsampling (strides) with every convolution later, in order to reduce over-fitting. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 98).

#### 4. Appropriate training data

I used the tract 1 training data provided in the course.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The model is derived from the nVidia model, discussed in the course. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training and validation set. But on running it with simulator, it would go off road. My interpretation was that the model was not generalized enough. It was fitting the training and validation set. But without enough generalization, just a few wrong predictions would make it go off the track.

To combat the overfitting, I modified the model to subsample and added dropouts. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 82-96) consisted of a convolution neural network with the following layers and layer sizes.

Cropping images to ignore top and bottom parts which are irrelevant.
```
model.add(Cropping2D(cropping=((75, 25), (0, 0))))
```

Convolution layers:
```
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Dropout(.1))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Dropout(.1))
model.add(Conv2D(48, (3, 3), strides=(2, 2), activation='relu'))
model.add(Dropout(.1))
model.add(Conv2D(64, (3, 3), activation='relu'))
```

Followed by fully connected layers.
```
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Dense(54))
model.add(Dense(1))
```

#### 3. Creation of the Training Set & Training Process

The track 1 training set has 8036 data points. 

To augment the data sat, I also flipped images and angles thinking that this would and randomly changed the brightness of the images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The model seems to converge within 3 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
