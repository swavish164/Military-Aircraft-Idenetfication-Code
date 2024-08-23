import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import os

airCraftTags = ['A10','A400M','AG600','An72','AV8B','B1','B2','B21','B52','Be200','C2','C5','C17','C130','C390','E2','E7','EF2000','F4','F14','F15','F16','F18','F22','F35','F117','H6','J10','J20','JAS39','JF17','JH7','KC135','KF21','KJ600','Mig31','Mirage2000','MQ9','P3','Rafale','RQ4','SR71','Su24','Su25','Su34','Su57','TB001','TB2','Tornado','Tu22M','Tu95','Tu160','U2','US2','V22','Vulcan','WZ7','XB70','Y20','YF23']


model = load_model('AircraftIdentification.keras')
input_size = 64

image= cv2.imread('Aircraft\JF17\\0fee0a284a76e8fdb30da0f26a44ab49_0.jpg')
img = Image.fromarray(image)

img = img.resize((input_size,input_size))
img= np.array(img)
input_imgs = np.expand_dims(img,axis=0,)

predictions = np.array(model.predict(input_imgs))
print(predictions)
value = np.where(predictions[0] == 1 )
print(airCraftTags[value[0][0]])