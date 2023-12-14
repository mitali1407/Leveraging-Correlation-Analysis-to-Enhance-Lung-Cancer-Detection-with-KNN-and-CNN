# Leveraging-Correlation-Analysis-to-Enhance-Lung-Cancer-Detection-with-KNN-and-CNN

Here in this study K-Nearest Neighbors (KNN) algorithm is used for Image Processing of images and implemented using Python.The images are compared with the data set that has both the cancer affected and unaffected images.The input images are compared to the images in the data set and whether the cancer is presence of cancer is found.The aim of this project is to study the technique and implement it and analyse the image using digital image processing.

Methodology
1.  Data Acquisition and Pre-processing  
1.1 Image Data set Collection
Obtain a data set of lung images. It must have images of several cases. The data set should ideally consist of high-resolution medical images such as CT scans or X-rays. Collaborate with healthcare institutions or research organizations to ensure access to appropriate data.

1.2 Data Pre-processing
Convert the images to a standardized format (e.g., DICOM).Apply image pre-processing techniques to enhance image quality, such as noise reduction, contrast adjustment, and image resizing.

1.3 Feature Engineering

Utilize digital image processing techniques to extract meaningful features from the pre-processed images. Common features may include Texture features (e.g., Haralick, Gabor, or LBP features) to capture textural patterns.

1.4 Feature Selection

Evaluate the extracted features and perform feature selection to retain the most relevant ones. Techniques like feature ranking or dimensionality reduction (e.g., PCA) can be applied to optimize feature sets.

1.5 Data Splitting

Split the data set .Ensure that an appropriate balance between cancerous and non-cancerous images is maintained in both sets to prevent class imbalance issues.

  
1.6 KNN Model Development

KNN Classifier Implementation: Implement the K-Nearest Neighbors (KNN) algorithm using a suitable machine learning library such as scikit-learn.

Hyper-parameter Tuning: Determine the optimal value of 'k' (number of neighbors) through cross-validation techniques (e.g., grid search) on the training data.

Model Training: Train the KNN model using the training set and the selected hyper-parameters.
 
Model Evaluation: Performance Metrics: Evaluate the KNN model's performance on the testing data set using various performance metrics.

Image Enhancement:It is the process of enhancing the quality of the image exampleimproving them visually, enlarging or bringing out special features, removing
noises etc.
Image Enhancement is classified into two categories :
Spatial Domain
Frequency Domain.

1.7 Median Filter
Also known as non-linear filtering, it is  a type of filter used to get rid of
salt and pepper noise which includes removing pixels with extreme values while preserving edges of digital images. It is also used to sharpen the
contrast. .It works by replacing each pixel's value with the median value of the pixels in its neighborhood.

1.8 Marker Controlled Watershed Segmentation
The segmentation techniques based on edge and region are part of the segmentation approach. One of these techniques is Thresholding. An image that has pixels in two values, 1 and 0, is called a binary image. The advantage of using Threshold Segmentation is that it requires less storage space and has faster speed compared to grayscale images

Watershed Transformation: It is used to segment an image based on the topography of the image intensity. The intensity values are represented as elevations, where bright regions are considered as high elevations, and dark levels are considered as low elevations.

Gradient Image: This is used to define the topography of the image, serving as the starting step for the segmentation process. Intensity transitions are highlighted in the original image.

Watershed Lines: These act as segmentation boundaries between regions.
Merge and Refinement: To avoid extra regions generated during segmentation, post-processing steps are followed to merge small regions that are not areas of interest.

It is the process of obtaining numeric values from given raw data and extracting useful characteristics that are important for image analysis. These features help generate patterns within an image. To obtain a broader range of image data, several feature extraction methods can be used together. The original information from the data set is preserved.

Applications of feature extraction include Edge Detection, Texture Analysis, Segmentation Features, etc.

 
1.9K-Nearest Neighbour:

K-Nearest Neighbors (KNN) is an algorithm that groups data based on similarity to already existing examples. Grouping is determined by the neighbors of a data point. KNN is dependent on the similarity of features, and it is abbreviated as KNN. Selecting the right value of K is important for accuracy. 
