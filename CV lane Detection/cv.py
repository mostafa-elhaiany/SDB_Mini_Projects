import cv2
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread('input images/0.png')
def detect(image):
  """
    Lane detection function
    INPUT: frame/image of a road
    OUTPUT: frame/image of a road with the lanes highlighted
  """
  #open cv reads the image in bgr format so we transform it into rgb (try using LAB,HSV, and HLS)
  image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
  height = 250
  width = 250
  #we resize the image into a fixed size 
  image = cv2.resize(image,(height,width)) 


  """
    for color detection usually people go for HSV as its easier to detect colours in that colour space
    for our approach since we're trying to detect the white colour RGB would work well,
    the lower bound for white is [0,190,0] while the upper bound is [255,255,255] we can use
    these bounds in the inRange function where any pixel that lies inside that range passes as a 
    true(1) value and every other pixel passes as false(0) value.
    Try playing around with the bounds and see the effects on the resulting image.
  """
  lower_bound = np.array([0,190,0])
  upper_bound = np.array([255,255,255])

  mask = cv2.inRange(image, lower_bound, upper_bound)
  #multiply the mask by the original image to get only the white pixels
  masked_image = cv2.bitwise_and(image,image, mask= mask)


  """
    By now we can see that some of the environment(cars, the sky, etc) also passes through our mask(if you 
    increased the lowerbounds maybe you'll see less objects in the image), some of these pixels are not 
    important to check in our algorithm since they lie outside of our Region of Interest (RoI)
    so the next snippet of code only takes into consideration a portion of the image, play around 
    with the vertices and see how they affect the detection of the line
  """
  vertices = np.array([[0,height], [width, height],
      [int(0.55*width), int(0.6*height)], [int(0.45*width), int(0.6*height)]])

  mask = np.zeros_like(masked_image)   
  channel_count = image.shape[2]
  ignore_mask_color = (255,) * channel_count
  cv2.fillPoly(mask, np.int32([vertices]), ignore_mask_color) #fills the area under the polygon specified by our verticies by ones
  masked_image = cv2.bitwise_and(masked_image, mask) #multiplies the original image by the mask generating the image with only the ROI


  """
    By now, we have, idealy, only the lanes detected in our image, the rest is to extract the edges and 
    draw the lines, this can be done on an rgb image, however, its time consuming and not worth it, since 
    we already extracted the colours we need we no longer need to go on with the coloured image
  """
  grayscale = cv2.cvtColor(masked_image,cv2.COLOR_RGB2GRAY)  # transforms the image into a grayscale image
  
  edges = cv2.Canny(grayscale, 50, 100)#detects the edges inside the grayscale image.
  #The two values are threshholds play around and see if they affect the results



  """
    now that we have all the edges present inside the grayscale image, its time to extract the lines, these lines should represent
    the lanes inside our original image.
    The next set of variables are values the function hough lines uses to draw the lines, you can read more about them in opencv docs,
    but by intuition they're the values that choose which lines are drawn and which ones are discarded.
  """
  rho= 1
  theta=np.pi/180
  threshold= 20
  min_line_len= 20
  max_line_gap= 250

  lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_len, max_line_gap) #extracts lines in image
  line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)#blank image to draw the lines on
  """
    each line is represented with two points, (x1,y1) to (x2,y2), this is exactly what the function returns,
    an array of arrays each sub array is an array of 4 values representing the 2 points needed to draw a line.
    Maybe print lines and see how they look if you need to
  """
  for line in lines:
      for x1,y1,x2,y2 in line:
          cv2.line(line_img,(x1,y1),(x2,y2),(0,255,0),4) 
          #drawing the line, takes the two points, a colour and thickness.


  """
    Finally we need to draw the lines on top of the original image, the next function does a weighted 
    addition of two images final = image1 * (value) + image2 * (1-value) where the value here corresponds to 0.8
  """
  final = cv2.addWeighted(image, 0.8, line_img, 1.0, 0) 
  return final

cap = cv2.VideoCapture('3.mp4') #reads video frames from a source
#the source can either be a path to a video saved localy, or a port that corresponds to a webcam for a live feed 

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

"""
  The next line of code is used to save a video after processing in python it creates a video writer that you can write frames to after
  you're done processing then save the resulting video, 
"""
# out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (250,250)) 

# Read until video is completed
while(cap.isOpened()):
  
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True: # ret is a boolean that return False if there was an error in reading the frame from cap

    # Display the resulting frame
    final = detect(frame)
    
    cv2.imshow('Frame',final) # this line shows the frame after preprocessing in a new cv2 window

    # cv2.imwrite('1.png',final) # this line if un commented would save every frame to an image called 1.png
    
    # out.write(frame) # this line would write frames to the video writer  

    # Press Q on keyboard to  exit
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break

  # Break the loop if ret is false
  else: 
    break

# When everything done, release the video capture object
# out.release() #remember to release the video capture so it doesn't result in a corrupt video
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

