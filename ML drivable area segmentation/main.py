import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import keras
import cv2
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

"""
In this file we're going to do a drivable area segmentation
Segmentation means that we do pixelwise classification, in this case, a binary one.
Every pixel will get classified into a drivable area(1) or a nondrivable area(0).

In machine learning, all layers created are created with random initialization, one of the most important steps is reproducability of 
results, so if i run the code locally you should get exactly the same exact results every run.
This is handled by setting a random seed.
"""
np.random.seed(42)


"""
first we need to read the input images and labels,
we used cv2.imread(IMAGE_PATH) to read one image from you local machines.
Glob returns a list of all the items in the specified path, so we get a list of all the paths belonging to the images inside 
the images and the labels folders
"""
X_train_paths = glob('images/*')
y_train_paths = glob('labels/*')

"""
we specify an image width and an image height, for segmentation is always easier to compute the input and outputs of layers if the 
image size is a power of 2
"""
width=256
height=256


'''
now that we have the image sizes and their paths, we can read them into an array of X and y like we did with cifar
cv2.cvtColor() makes sure that the input images are rgb, and the labels are grayscaled,
play around with the bgr2rgb and see if you can get better results by choosing a different color space.
cv2.resize() resizing the image to our predefined width and height
cv2.imread() finally reads the input images
'''
x_train=[]
y_train=[]
for idx,path in enumerate(X_train_paths):
        x_train.append(cv2.cvtColor(cv2.resize(cv2.imread(path),(width,height)),cv2.COLOR_BGR2RGB))
        y_train.append(cv2.cvtColor(cv2.resize(cv2.imread(y_train_paths[idx]),(width,height)),cv2.COLOR_BGR2GRAY))


"""
then we vectorize the input from being a python list to a numpy array.
"""
x_train = np.array(x_train)
y_train = np.array(y_train)


# #uncomment this part to see a representation of the input image
# plt.figure()
# plt.imshow(x_train[0])


'''
since the labels are grayscaled cv2 saves the image with shape (width,heigh) we need to make sure that it saves the channels as well
so the next lines saves the image with shape (width, height, 1)
this isnt needed if the image has more than one channel
'''
y_train = np.expand_dims(y_train,axis=-1) 

# print the input and out put shape to see if everything is right
print(x_train.shape)
print(y_train.shape)



'''
same as we did with cifar we create our segmentation model.

Segmentation models consists of two main parts 
the Encoding and the Decoding

Encoding blocks are convolution + maxpooling, they encode the image by reducing its dimensionality 
Decoding blocks are Upsample + convolution, they decode the image by increasing its dimensionality

'''

model = keras.models.Sequential()
#input --------------> outputs a feature map (height,width,channels) = (256,256,128)
model.add(keras.layers.Conv2D(128,3,padding='same',input_shape=(height,width,3),activation='relu'))

#encoding block 1---------->outputs a feature map (height,width,channels) = (128,128,64)
model.add(keras.layers.Conv2D(64,3,padding='same',activation='relu'))
model.add(keras.layers.MaxPooling2D(2))

#encoding block 2---------->outputs a feature map (height,width,channels) = (64,64,64)
model.add(keras.layers.Conv2D(64,3,padding='same',activation='relu'))
model.add(keras.layers.MaxPooling2D(2))

#encoding block 3---------->outputs a feature map (height,width,channels) = (32,32,32)
model.add(keras.layers.Conv2D(32,3,padding='same',activation='relu'))
model.add(keras.layers.MaxPooling2D(2))

#decoding block 1---------->outputs a feature map (height,width,channels) = (64,64,64)
model.add(keras.layers.UpSampling2D())
model.add(keras.layers.Conv2D(64,3,padding='same',activation='relu'))

#decoding block 2---------->outputs a feature map (height,width,channels) = (128,128,128)
model.add(keras.layers.UpSampling2D())
model.add(keras.layers.Conv2D(128,3,padding='same',activation='relu'))

#decoding block 3---------->outputs a feature map (height,width,channels) = (256,256,1)
model.add(keras.layers.UpSampling2D())
model.add(keras.layers.Conv2D(1,3,padding='same',activation='sigmoid'))

'''
notice that for every encoding block there is a corresponding decoding block to get the same size input as output
in the session we previewed intermediete convolutuion results by making a layer with filtersize 3 and showing this using 
matplotlib.pyplot. maybe play around and try to get it.
'''


model.summary() #this summarizes the model layers and parameters

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


y_train[y_train>0] = 1 #this makes sure that the labels consist of values 0 and 1 only to binarize the output

x_train = x_train.astype(np.float32) / 127.5  - 1 #ML models usually do better with normalized inputs so we normalize the input from -1 to 1


model.fit(x_train,y_train,epochs=10,batch_size=16,validation_split=0.1) # trains the model 

model.save('segment.h5') #saves the resulting model

model = keras.models.load_model('segment.h5') #loads the saved model

'''
now we need to test if the model actually leanred and is predicting nicely 
first we use predict() to predict on input images
'''
preds= model.predict(x_train[:2])
 #we make sure that the output prediction is binarized(0 and 1 only)
preds[preds<0.5]=0
preds[preds>=0.5]=1


'''
plotting the results as input label and prediction
'''
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.imshow((x_train[0]+1)/2)
ax2.imshow(y_train[0][:,:,0])
ax3.imshow(preds[0][:,:,0])
plt.show()

