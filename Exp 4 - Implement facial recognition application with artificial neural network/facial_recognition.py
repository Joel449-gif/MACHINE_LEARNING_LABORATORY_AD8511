
TrainingImagePath='E://Sem 5//ML//ML_Lab//exp4//Face Images//Final Training Images'
 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
 
test_datagen = ImageDataGenerator()

training_set = train_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
 
test_set = test_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
 
print(test_set.class_indices)

TrainClasses=training_set.class_indices

ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[faceValue]=faceName

import pickle
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)
 
print("\n Mapping of Face and its ID",ResultMap)
 
OutputNeurons=len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
 
classifier= Sequential()
 
classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))
 
classifier.add(MaxPool2D(pool_size=(2,2)))

classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
 
classifier.add(MaxPool2D(pool_size=(2,2)))
 
classifier.add(Flatten())
 
classifier.add(Dense(64, activation='relu'))
 
classifier.add(Dense(OutputNeurons, activation='softmax'))
 
classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])

import time

StartTime=time.time()
 
classifier.fit_generator(
                    training_set,
                    steps_per_epoch=30,
                    epochs=10,
                    validation_data=test_set,
                    validation_steps=10)
 
EndTime=time.time()
print("###### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ######')

import numpy as np
import keras
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
 
ImagePath='E://Sem 5//ML//ML_Lab//exp4//Face Images//Final Testing Images//face4//3face4.jpg'
test_image= keras.utils.load_img(ImagePath,target_size=(64, 64))
test_image=image.img_to_array(test_image)
 
test_image=np.expand_dims(test_image,axis=0)
 
result=classifier.predict(test_image,verbose=0)
 
print('####'*10)
print('Prediction is: ',ResultMap[np.argmax(result)])