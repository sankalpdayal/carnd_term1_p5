** Vehicle Detection Project **

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images
* Also apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Normalize features and randomize a selection for training and testing.
* Train a classifier like Linear SVM classifier, SVM with Gaussian Kernel or Decision Trees.
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run  pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/car_not_car_hog.jpg
[image3]: ./examples/window_search1.jpg
[image4]: ./examples/window_search2.jpg
[image5]: ./examples/output_bboxes.png
[image6]: ./examples/bboxes_and_heat.png
[video1]: ./project_video_output.mp4
[video1]: ./project_video_output_with_hist.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. This is the writeup that includes all the rubric points and how each one was addressed. 

### Data Exploration

First thing I did was to get understanding of size of data, shape if image, datatype of image. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

### Histogram of Oriented Gradients (HOG)

#### 1. Extraction of HOG and other features from the training images.

The code for this step is contained in the third code cell of the IPython notebook 'Vehicle_Detection.ipynb' under the title 'Method to get histogram of gradients'. The function `get_hog_features` takes in the image and parameters like orient, pixles per cell etc and returns features. It optionally
also returns HOG feature image. 

Here is an example using the B channel of `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)`:

![alt text][image2]

To extract color histogram I used the function `color_hist`. This function is under the title 'Method to extract color historgram'. The function takes in image, number of bins and bins range as input and outputs features.

To extract color histogram I used the function `bin_spatial`. This function is under the title 'Method to extract binned color features'. The function takes in image and binning size as input and returns features.

To extract HOG, color histogram and bin features all under one function, I created a function `extract_features` contained in the the cell under title 'Combined Function to extract features from a list of images'. The function takes list of image addresses, a dictonary stating what features, color space, and parameters for HOG, color histogram and binned color features.
Dictonary stating what features has 3 keys 'spatial', 'hist', 'hog' and their values can be True or False depending on if that feature has to be extracted or not. The function does teh following

1. Loop over the image addresses and load each image.
2. Change the color space of image depending on the input.
3. Extract features depending on key values in the extract_features dictonrary.
4. For HOG, if the channels are 'ALL' then extract HOG features for all the three channels and concatenate them.
5. Finally append all the features extracted and return these.

#### 2.Final choice of HOG parameters.

There can be so many combinations for color space, spatial bins size, orientations etc. To identify best combination I created a list of multiple combinations to choose from. The exploration space had following options

1. color spaces
2. possible combination of features
3. spatial bins
4. histogram bins
5. orienations
6. pixels per cell
7. cells per block and hog channels.

For all possible values of options I created a all possible combinations. The code that does that is defined under the cell titled 'Create all possible features combinations'. The code creates a dataframe that has all the combinations. I tried 432 combinations 

For each combination, features were extracted accordingly, then these features were normalized and then a Linear SVM was trained. Accuracy of the trained Linear SVM was tested on test data. The combination which showed the best validation accuracy was stored. 
The code that does this in following steps

1. Current combination in the search is loaded
2. Depending on the current combinations, features are extracted using the `extract_features` function.
3. Features are normalized using `StandardScaler()` method.
4. Data is splitted into train and test. Same random state is used for each combination.
5. Linear SVM with hinge loss is trained on train data and tested on test data
6. Model with best accuracy is stored.

The feature selection which showed the best accuracy has following combination.

|Option|Value|
|:----|:----|
|color_spaces  |HSV|
|hist |True| 
|spatial  |True|
|hog |True|
|histbins |16|   
|spatialbins  |16|
|cells_per_block |2|
|hog_channels  |All|
|orient  |11|
|pixels_per_cell |8|

#### 3. Training a classifier using selected HOG features and color features

Since the feature selection and classifer training was combined in previous step, the combination which showed best accuracy resulted in 'LinearSVM' calssifier. The calssifier has validation accuracy of '0.989' and training time of 4.43 seconds. 

Using the same set of features, Decision Trees and SVM with Gaussian Kernel was also trained. Their validation accuracy was 0.9505 and 0.9927 respectively. 

Although finally LinearSVM was used because decision tree had lower accuracy and Gaussian Kernal had high predicition time.


### Sliding Window Search

#### 1. Describption ofimplementation of sliding window search

The function to define the search space is given in the cell in python notebook under the title 'Define region for search using the sliding window approach'. I used the test images to decide what the positions and scales. Idea was to make sure all the cars
in the test images are covered and no search is going on in the areas where there are no cars. I tried scales from 1 to 3. Some examples of the sliding window search are given here.

![alt text][image3]

![alt text][image4]

#### 2. Vehicle Detection Pipeline (For Image)

The search and detection process of combined under a single function `find_cars()`. The function takes in the image, search space combination (y start, y stop, scale, overalp) and feature combination. It also takes 2 classifiers and input. Positive detection is considered if both classifiers 
find a true detection in the search window. Find cars function implements steps to find car using the sliding window approach

1. Using the y start, y stop and scale, defines the search space using the sliding window approach.
2. Extract features depending on the input parameters given for window current being searched
2. Normalize the extracted features
3. Use the normalized features for prediction using and of two classifiers
4. If prediction from both is true, then store the current window in the rectangles

Ultimately I searched on 5 scales and each scale had different Y start and stop positions and overlap. Scales which were smaller were restricted to regions further from car. This is because in those regions car will smaller in size. For higher scales, tend to be more closer
to car. Also these had higher overlap values to keep the step size smaller. Finally following scale and y start and y stop values and overlap are used

| Scale         | Y start       | Y Stop  |Overlap| Reason  |
|:-------------:|:-------------:|:-------:|:-----:|:--------|
|1.0|380|500|0.5|Detect far small cars|
|1.5|380|520|0.75|Detect near small cars|
|1.75|360|530|0.75|Detect near medium sized cars|
|2.8|360|550|0.75|Detect near medium sized cars|
|3.3|360|580|0.75|Detect near large sized cars| 

Vehicle Detection Pipeline was combined under a single function `process_image(test_img)`. This function is contained in the cell under the title 'Process image implements the car finding pipeline in an image'. The function implements following steps

1. Load the classifiers and parameters for feature extraction.
2. Call find cars function for different scale and y start and y stop values.
3. Append rectanges from all the find car calls.
4. Create a single list from all the aboved appended list and return this single list

As stated before, both classifiers were same and were LinearSVM because they give good accuracy (compared to decision trees) and faster prediction (compared to SVM with Gaussian Kernel). The pipeline was tested on test images. The results are shown below

![alt text][image5]

To reduce false positives, method of creating a heatmap and thresholding the heatmap was used. The functions are same as taught in the classroom and are given under the cell 'Create heatmap to reduce false positives'. `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap was used. Each blob is assumed to corresponded to a vehicle.  
Bounding boxes are constructed to cover the area of each blob detected. The result after thresholding on the same images, are shown below.

![alt text][image6]

---

### Video Implementation

#### 1. If the same pipeline is used on a video. The vehicle detection bound box comes out somewhat wobbly or unstable bounding boxes and some false positives. The output is shown here 

[![Vehicle Detection (Frame by frame)](http://img.youtube.com/vi/cRGjzo0Qmyc/0.jpg)](https://www.youtube.com/watch?v=cRGjzo0Qmyc)

To make the bounding box more stabe and fewer false positives, time based information can be used. This means instead of doing fresh detection everytime with new frame, some history can be used. I have used a clustering method to do this. The output from this technique is shown here 

[![Vehicle Detection with Clustering](http://img.youtube.com/vi/N-WdC5BG8eU/0.jpg)](https://www.youtube.com/watch?v=N-WdC5BG8eU)

#### 2. Description of clustering technique to filter for false positives and method for combining overlapping bounding boxes.

To real life scenario following assumptions can be made

1. Vehicles if visible can last for few seconds in the camera.
2. The position of the vehicle (in the frames) will move but slowly with time.
3. If there is a vehcile, there will be positive detections from classifier not only in a single frame but also accross multiple frames.
4. Since The position of the vehicle (in the frames) will move but slowly with time, the size of the bounding box will also increase or decrease slowly.

Considering these assumpitions I implemented a clustering technique to achieve desired result. The technique works in the following manner

1. It consideres the latest 30 frames (1.5 seconds)
2. For these latest frames, it iterates over all the positive detections. 

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.   

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

