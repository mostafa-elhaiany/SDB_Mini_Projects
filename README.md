# SDB_Mini_Projects
Self Driving Bike research cluster mini projects

## Getting Started

This repo has mini projects used in the sessions of the sdb research cluster

### Prerequisites

The code runs on python 3.7
you'll need the following libraries

```
Numpy
```
Adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

```
Opencv
```
This is the library used for real-time computer vision task.

```
Matplotlib 
```
This is a plotting library for Python and is used with NumPy.


```
keras, along with tensorflow backend
```
Keras is an open-source library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library

### Installing



make sure you have a python3.7 setup up and running on your machine
you can install anaconda and create a new environment using the following link
```
https://www.geeksforgeeks.org/set-up-virtual-environment-for-python-using-anaconda/
```

then to install the needed libraries, activate your new environment and use the following commands

```
pip install numpy opencv-python matplotlib
```

```
pip install tensorflow==2.1.0
```

```
pip install keras==2.3.1
```



to make sure everything is up and running, run a python script including the imports of the beforementioned libraries.


### Break down into file system and Algorithms used

Every directory holds the code used for each session given along with its documentation

```
CV lane Detection
```
Holds The basic lane detection code, the pipeline used in this code is as follows 
                1) Turn image into RGB colourspace <br/>
                2) Resize image into a fixed width and height <br/>
                3) Use a mask to detect the white colour inside the frame <br/>
                4) Extract the region of interest of the frame <br/>
                5) Convert to grayscale <br/>
                6) Extract edges  <br/>
                7) Extract Lines <br/>
                8) Generate Image with only the lines present in the frame <br/>
                9) Generate the final outcome with the lanes highlighted on the original frame <br/>


### Running the code

inside each directory run main.py





