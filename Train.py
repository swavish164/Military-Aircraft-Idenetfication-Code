import cv2
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import utils
from tensorflow.keras.utils import normalize 
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D,Activation,Dropout,Flatten,Dense, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras.utils import to_categorical
import pickle
import random 


image_directory = 'Aircraft/'
dataSet = []
labels = []

airCraftTags = ['A10','A400M','AV8B','B1','B2','B52','Be200','C2','C17','C390','E7','EF2000','F4','F15','F16','F117','J10','J20','JAS39','JF17','Mirage2000','SR71','Su25','Su57','Tornado','Tu95','U2','Vulcan','XB70','YF23']


#tumor_images = os.listdir(image_directory+'Brain Tumor')
#healthy_images = os.listdir(image_directory+'Healthy')
input_size = 64
"""
def preprocess(images,tag):
  for i, image in enumerate(images):
    if(image.split('.')[1]=='jpg' or image.split('.')[1]=='JPG'):
      image_path = image_directory+tag+image
      image_array = cv2.imread(image_path)
      image_array = Image.fromarray(image_array,'RGB')
      image_array = image_array.resize((input_size,input_size))
      dataSet.append(np.array(image_array))
      labels.append(airCraftTags.index(tag[:len(tag)-1]))
        
        
for tags in airCraftTags:
    images = os.listdir(image_directory+tags)
    preprocess(images,tags+"/")
    
print(len(dataSet))
dataSet = np.array(dataSet)
labels = np.array(labels)
with open("dataArray.pkl","wb") as f:
  pickle.dump(dataSet,f)
with open("labels.pkl","wb") as f:
  pickle.dump(labels,f)
  
  
"""

with open("dataArray.pkl", "rb") as f:
  dataSet = pickle.load(f)
with open("labels.pkl", "rb") as f:
  labels = pickle.load(f)
  
combined = list(zip(dataSet, labels))
random.shuffle(combined)
dataSet, labels = zip(*combined)


x_train = dataSet[int(len(dataSet)*0.2):]
x_test = dataSet[:int(len(dataSet)*0.2)]
y_train = labels[int(len(labels)*0.2):]
y_test = labels[:int(len(labels)*0.2)]

x_train = normalize(x_train,axis = 1)
x_test = normalize(x_test,axis = 1)

y_train = to_categorical(y_train,num_classes = 60)
y_test = to_categorical(y_test, num_classes = 60)


model = Sequential()
#model.add(Input(shape=(input_size,input_size,3)))
# First Convolutional Block
model.add(Conv2D(64, (3, 3),padding = 'same', input_shape=(input_size, input_size, 3), kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Block
model.add(Conv2D(128, (3, 3),padding = 'same', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Convolutional Block
model.add(Conv2D(256, (3, 3),padding = 'same', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fourth Convolutional Block
model.add(Conv2D(512, (3, 3),padding = 'same', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fifth Convolutional Block (optional, for more capacity)
#model.add(Conv2D(512, (3, 3),padding = 'same', kernel_initializer='he_uniform'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the network
model.add(Flatten())

# Fully Connected Layer 1
model.add(Dense(512, kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Fully Connected Layer 2
model.add(Dense(256, kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Fully Connected Layer 3 (optional)
model.add(Dense(128, kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(60))
model.add(Activation('softmax'))

datagen = ImageDataGenerator(rotation_range = 20, width_shift_range = 0.2,height_shift_range = 0.2, shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True,fill_mode = 'nearest')
optimizer = Adam(learning_rate = 0.0001)
early_stopping = EarlyStopping(monitor = 'val_loss',patience = 10)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',factor = 0.5,patience = 5,min_lr = 0.00001)

checkpoint = keras.callbacks.ModelCheckpoint('model_checkpoint.keras', save_best_only=True, save_freq='epoch', monitor='val_accuracy', mode='max')
model.compile(loss = 'categorical_crossentropy',optimizer = optimizer,metrics = ['accuracy'])
model.fit(x_train,y_train,batch_size = 16,verbose = 1,epochs = 50,validation_data= (x_test,y_test),shuffle = True)
#model.fit(datagen.flow(x_train, y_train, batch_size=16), steps_per_epoch=len(x_train) // 16, epochs=50, validation_data=(x_test, y_test),shuffle = True,callbacks = [checkpoint])
model.save('AircraftIdentification.keras')

