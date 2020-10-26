import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
'''
importing keras and tensorflow results in tensorflow trying to activate cuda and run with gpu this usually leads to a lot of logging 
done on the part of tensorflow that we simply might not want to see, the above two lines makes sure the tf logging is only set to 
errors.

The following is the two libraries we will need to import keras for the ML and dataset, and matplotlib for plotting functionalities.
'''
import keras
import matplotlib.pyplot as plt
import numpy as np

'''
The dataset we are using is the cifar10 dataset a famous benchmark dataset that is usually used for testing.
There is also cifar100 and cifar1000 or a more easy one like mnist. 

The cifar 10 dataset consists of the following 10 classes: 
        1-airplane
        2-automobile
        3-bird
        4-cat
        5-deer
        6-dog
        7-frog
        8-horse
        9-ship
        10-truck
We would load this dataset in order to train an ML model.
keras already has pre-defined datasets that we can import. They return 4 arrays
        1-X_train: These are the pictures that are used for Training
        2_y_train: Each image in the x_train needs a corresponding label in order to learn to predict correctly

        3-X_test: After the model is trained a set of images that are not present in X_train are used for testing
        4-y_test: Each of the images in X_test have a corresponding label in order to get a metric of how good the model is.
So (X,Y)_train are used to train the model, which means the model would learn(edit its parameters) according to the datapoints
present in the training set. Afterwards, the model is tested on an unseen data to show a final evaluation of how good the model has
learned.
'''
(X_train,y_train),(X_test,y_test) =keras.datasets.cifar10.load_data() 


'''
After reading the dataset, let's see if we can show some of the data using matplotlib, but first, since the labels inside y_train and 
y_test are numbers lets use the following dict in order to get a better understanding of the printed values.
'''
class_dict={
  0:"airplane",
  1:"automobile",
  2:"bird",
  3:"cat",
  4:"deer",
  5:"dog",
  6:"frog",
  7:"horse",
  8:"ship",
  9:"truck"  
}
ix = np.random.randint(0,len(X_train),16) #select random 16 values from training dataset
t_x = X_train[ix]
t_y = y_train[ix]
t_y = np.array([class_dict[i[0]] for i in t_y]) #set the values in the array to the string representation 

fig, m_axs = plt.subplots(4, 4, figsize = (10, 10)) #create an empty figure with 16 subplots
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()): #for every subplot get an image, its corresponding label and the axis to draw on
    c_ax.imshow(c_x) # show the image
    c_ax.set_title(c_y) #show the label above the image
    c_ax.axis('off') #remove the axis values
plt.show() #show the plot

'''
The above snippet of code should result in a 4x4 grid of images with their corresponding labels.
So now we have a feel of what our task is. The task is to classify the data.
In order to do that we need to create our model. Keras has two types of models we could build.
For this task we will be using Sequential models(maybe try to find out what the other is? we said it in the session).
To create a sequential model we can think of it as an array of layers, each layer takes its input from the layer that perceeds it.
'''
model = keras.models.Sequential() # first we create our empty model


"""
we discussed three layers that we will be using in this task
CONV layers:
These will be the filters that are convoluted over the image producing feature maps 
so for an image of size 32x32 as the one we have, a conv layer will decrease its dimensions but add more depth to the image according
to the number of filters we use.

POOLING layers:
Pooling layers are used to progressively reduce the spatial size of the image or feature maps and to reduce the amount of
parameters and computation in the network. The most common pool size is (2,2) which decreases the image to half its size
(what happens if i use other sizes? (3,3),(4,4),(3,5)?)


FLATTEN layers:
as the name suggests these layers flatten the input into a 1 dimensional array of features that can be used as an input to a Dense layer
ex:
[
    [1,2,3],
    [4,5,6],  ------------->    [1,2,3,4,5,6,7,8,9]
    [7,8,9]
]

DENSE layers:
These layers are the fully connected layers present in an MLP
"""


model.add(keras.layers.Conv2D(64, 3, strides=(2,2), padding='same', input_shape=(32,32,3),activation='relu'))
'''
For the first layer we must always specify an input shape, it corresponds to the shape of one datapoint inside our dataset.
For our data we can see that the images are 32X32 in HxW and they have 3 channels corresponding to rgb.
We specify a filter size of 64 which means we will be having a feature map output with 64 new channels, a kernel size of shape (3,3)
and a strides of (2,2) which means we will half the image in size. An activtaion of relu is commonly used in the hidden layers of 
most models. (what other activation functions are there?)

For the rest of the model add Conv and maxpooling layers with the parameters you need I will choose the following.
Just make sure you do not decrease the dimension of the image to a negative value, keras will throw an error at you!
'''
model.add(keras.layers.Conv2D(64, 3, strides=(2,2), padding='same',activation='relu')) 
model.add(keras.layers.MaxPool2D(2))
model.add(keras.layers.Conv2D(32, 3, strides=(2,2), padding='same',activation='relu'))

'''
After we're done with our conv and pooling layers its about time we built the mlp responsible for getting us our output.
First, we flatten the feature map then use Dense layers!
'''
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
"""
dense layers take as parameters the number of neurons in the layer and an activation function
here we see the softmax activation function this function is used for multi class problems it makes sure that the output of the layer
corresponds to a probabilty of each class, which means the sum of all the values in the layer after should equal to 1.
what happens if I do a softmax activation with 1 layer? what other activation can I use for binary classification?
"""
model.summary() #this prints out a summary of the built model



"""
Now that we built the model, we need to compile it. Compiling a model means we specify the loss and the optimizer.
The loss is how far away are we from predicting the right answer, and the optimizer is how we use that metric to update our parameters.
In most classification problems an adam opimizer is used, adam stands for adaptive momentum(what is momentum? what other optimizers 
can I use?)
For the loss, a multi class problem is usually solved with a categorical crossentropy loss. This simply means that the labels are
one hot encoded. A one hot encoding of our labels would be the following:
0 ---> [1,0,0,0,0,0,0,0,0,0]
1 ---> [0,1,0,0,0,0,0,0,0,0]
2 ---> [0,0,1,0,0,0,0,0,0,0]
3 ---> [0,0,0,1,0,0,0,0,0,0]
4 ---> [0,0,0,0,1,0,0,0,0,0]
5 ---> [0,0,0,0,0,1,0,0,0,0]
6 ---> [0,0,0,0,0,0,1,0,0,0]
7 ---> [0,0,0,0,0,0,0,1,0,0]
8 ---> [0,0,0,0,0,0,0,0,1,0]
9 ---> [0,0,0,0,0,0,0,0,0,1]
a function we can use to turn our labels from numerical to one hot encoding without doing it manually is:
y = keras.utils.to_categorical(y,num_classes)
However, keras has another loss function called sparse categorical crossentropy which handles the input as a sparse numerical value
and calculates the categorical loss function for you, we will be using that one for this task
"""
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

'''
Finally we need to train our model. 
Training the model is done with the fit function
the fit function takes the (X,y)_train and trains for e epochs, the epochs are the number of iterations where the model is trained
over the entire training data, the batch size dictates how many times the parameters are changed in one epoch.
'''
model.fit(X_train,y_train,epochs=20,batch_size = 64,validation_split=0.02) 

loss,acc=model.evaluate(X_test,y_test) # this function is used to evaluate the trained model on the testing data.
print('evaluation loss',loss)
print('evaluation accuracy',acc)


# recal the plotting we used before?
ix = np.random.randint(0,len(X_test),8) #select random 16 values from TESTING dataset
t_x = X_test[ix]
t_y = y_test[ix]
t_p = model.predict(t_x) # get the model's predictions
t_y = np.array([class_dict[i[0]] for i in t_y]) 
t_p = np.array([class_dict[np.argmax(i)] for i in t_p]) # the argmax function returns the predicted class by getting the index with the highest confidence level
fig, m_axs = plt.subplots(4,2, figsize = (10, 10)) 
for (c_x, c_y,c_p, c_ax) in zip(t_x, t_y,t_p, m_axs.flatten()): 
    c_ax.imshow(c_x) # show the image
    c_ax.set_title("label: "+c_y+"\nPred: "+c_p) #show the label above the image
    c_ax.axis('off') #remove the axis values
plt.show() #show the plot


"""
And this concludes how to build a classification CNN 
Can you build a better architecture of our model? What would you change?
"""