# DISCONTINUATION OF PROJECT #
This project will no longer be maintained by Intel.
Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.
Intel no longer accepts patches to this project.
# Motor Defect Detector


| Details            |                 |
|-----------------------|------------------|
| Target OS:            |  Ubuntu\* 16.04 LTS     |
| Programming Language  |  Python |
| Time to complete:     |  30 min      |


## What it Does
The monitoring of manufacturing equipment is vital to any industrial process. Sometimes it is critical that equipment be monitored in real-time for faults and anomalies to prevent damage and correlate equipment behavior faults to production line issues. Fault detection is the pre-cursor to predictive maintenance.

This reference implementation covers the basic implementation of FFT, Logistic Regression, K-Means clustering, GMM. It also shows how FFT is helpful in Feature engineering of vibrational data of a machine.  

## How it Works
There are several methods which don&#39;t require training of a neural network to be able to detect failures, starting with the most basic (FFT), to the most complex (Gaussian Mixture Model). These have the advantage of being able to be re-used with minor modifications on different data streams, and don&#39;t require a lot of known previously classified data (unlike neural nets). In fact, some of these methods can be used to classify data in order to train DNNs.

## Requirements
### Hardware Requirements
* Tested on the [TANK AIoT Dev Kit](https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit)
 
### Software Requirements
- Ubuntu* 16.04
- Python 3.5 with the following libraries: 
  * numpy:1.14.0
  * pandas:0.22.0
  * matplotlib.pyplot:2.1.2
  * sklearn:0.19.1
  * scipy:1.1.0
  * python-setup tools

## Setup

1. Download the Bearing Data Set at [https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
2. Extract the zip format of data into respective folder named 1st\_test/2nd\_test.
3. Make sure you have the following libraries:

  * numpy:1.14.0
  * pandas:0.22.0
  * matplotlib.pyplot:2.1.2
  * sklearn:0.19.1
  * scipy:1.1.0

4. Download the code.
5. Make sure all the code and data are in one directory.

## Run the application

Open the example in console or any python supported IDE (for example Spyder). Set the working directory where your code and dataset is stored.

 There are 4 samples: FFT, Logistic Regression, Kmeans, GMM. For each one of these, one or multiple files are available to execute.

| SAMPLE FILE NAME | EXPECTED OUTPUT | Note |
| --- | --- | --- |
| FreqTime.py | Frequency v/s Time plot |   |
| Utils.py | Have all the functions for all the modules to calculate amplitude. |  |
| Train\_logistic\_regression.py | Saved weight of the trained module in the filename &quot;LogisticRegression&quot; | Training for the logistic regression is done on testset1, bearing4\_y axis (failed), bearing 2\_x axis (passed) and testset2 bearing1 (failed), bearing2 (passed). *The training dataset is increased for a better result. |
| Test\_logistic\_regression.py | Result for the particular test set which bearing is failed and the plot of last 100 predicted labels for all the bearings. | Take the input path from the user, to check which bearing is suspected to fail and which is in normal condition. |
| Train\_kmeans.py | Saved weight of the trained module in the filename &quot;Kmeans&quot; | Training for the Kmeans is done on testset1, bearing4_y axis (failed), bearing 2_x axis (passed) and testset2 bearing1 (failed), bearing2 (passed). *The training dataset is increased for a better result.|
| Test\_kmeans.py | Result for the particular test set which bearing is failed and the plot of last 100 predicted labels for all the bearings. | Take the input path from the user to check which bearing is suspected to fail and which is in normal condition. The label 0-7 is the range (early-most critical for failure). It is able to detect the bearing 1_y axis which is very close to failure. |
| Train\_GMM.py | Saved weight of the trained module in the filename &quot;GMM&quot; | Training for the GMM is done on testset1, bearing4\_y axis (failed), bearing 2\_x axis (passed) and testset2 bearing1 (failed), bearing2 (passed). *The training dataset is increased for a better result. |
| Test\_GMM.py | Result for the particular test set which bearing is failed and the plot of last 100 predicted labels for all the bearings. | Take the input path from the user, to check which bearing is suspected to fail, and which is in normal condition. The label **0-2 is the range (early-most critical for failure)**. |

**Understanding the analysis methods**

**1. FFT:** A fast Fourier transform (FFT) is an algorithm that samples a signal over a period of time (or space) and divides it into its frequency components. These components are single sinusoidal oscillations at distinct frequencies each with their own amplitude and phase.

[Y](https://in.mathworks.com/help/matlab/ref/fft.html#f83-998360-Y) = fft([X](https://in.mathworks.com/help/matlab/ref/fft.html#f83-998360-X)) computes the discrete Fourier transform (DFT) of X using a fast Fourier transform (FFT) algorithm. If X is a vector, then fft(X) returns the Fourier transform of the vector. [**More details**](https://en.wikipedia.org/wiki/Fast_Fourier_transform)

**2. Logistic Regression:** Logistic regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes).

In logistic regression, the dependent variable is binary or dichotomous, i.e. it only contains data coded as 1 (TRUE, success, pregnant, etc.) or 0 (FALSE, failure, non-pregnant, etc.). The goal of logistic regression is to find the best fitting (yet biologically reasonable) model to describe the relationship between the dichotomous characteristic of interest (dependent variable = response or outcome variable) and a set of independent (predictor or explanatory) variables. [**More details**](https://en.wikipedia.org/wiki/Logistic_regression)

**3. Kmeans Clustering:** K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity.

 The results of the K-means clustering algorithm are:

- The centroids of the K clusters, which can be used to label new data
- Labels for the training data (each data point is assigned to a single cluster)

**Each centroid of a cluster is a collection of feature values which define the resulting groups. Examining the centroid feature weights can be used to qualitatively interpret what kind of group each cluster represents.** [**More details**](https://en.wikipedia.org/wiki/K-means_clustering)

**4. GMM:** A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. One can think of mixture models as generalizing k-means clustering to incorporate information about the covariance structure of the data as well as the centers of the latent Gaussians.

The GaussianMixture object implements the expectation-maximization (EM) algorithm for fitting mixture-of-Gaussian models. It can also draw confidence ellipsoids for multivariate models, and compute the Bayesian Information Criterion to assess the number of clusters in the data. A GaussianMixture.fit method is provided that learns a Gaussian Mixture Model from train data. Given test data, it can assign to each sample the Gaussian it mostly probably belongs to using the GaussianMixture.predict method.

The GaussianMixture comes with different options to constrain the covariance of the difference classes estimated: spherical, diagonal, tied or full covariance. [**More details**](https://en.wikipedia.org/wiki/Mixture_model)

**Code Explanation:**

For all the samples the basic approach is same. Following are the steps that are basic steps:

- Take the fft of each bearing of each file.
- Calculate the frequency and amplitude of it.
- Calculate the top 5 amplitude and their corresponding frequency.
- Repeat the same for each bearing and each data file. Stored the result in the result data frame.

**For FFT** : Gives the frequency vs time plot of each maximum frequency for each dataset.

- Copy the dataset to FFT/<br />
    cd path_to_FFT <br />
    cp -r path_to_1st_test .<br />
    cp -r path_to_2nd_test .<br />

- Execute FreqTime.py<br />
    python3 FreqTime.py<br />
    <br />
    enter the complete directory path to 1st_test/2nd_test: <I>path_to_FFT/1st_test/</I><br />
    enter the folder name 1st_test/2nd_test: <I>1st_test/</I><br />
    enter the max type in range 1-5 <I>value_from_1_to_5</I><br />

![Figure 1](./images/FFT/testset2/max2.jpg)  
*Figure 1. Plot for the testset2, max2 frequency for all the bearing.*

**For Logistic regression:**

Calculate the label for the data frame (in depended variable). Assume that first 70% is in normal condition and rest 71-100% are suspected to fail. For the passed one give label &#39;0&#39; and for suspected to fail give label &#39;1&#39;. Train the logistic regression on result data frame. Stored the model using numpy.

For testing: Take the input path from the user, to check which bearing is suspected to fail and which is in normal condition. Check the number of one label, and number of zero label in the last 100 predictions. If the number of one prediction is more than the number of zero label, then the bearing is suspected to fail. Otherwise, it is in normal condition.

- Copy the dataset to Logistic_Regression/<br />
    cd path_to_Logistic_Regression<br />
    cp -r path_to_1st_test .<br />
    cp -r path_to_2nd_test .<br />

- Train Logistic regression:<br />
    python3 LogisticRegressionTraining.py <br />
    <br />
    enter the complete directory path for the testset1 <I>path_to_Logistic_Regression/1st_test/</I><br />
    enter the complete directory path for the testset2 <I>path_to_Logistic_Regression/2nd_test/</I><br />

- Test the model<br />
    python3 logisticregressiontesting.py <br />
    <br />
    enter the complete directory path <I>path_to_Logistic_Regression/1st_test/</I><br />
    enter the folder name <I>1st_test/</I><br />

 ![Figure 2](./images/LogisticRegression/testset2_figure.jpg)  
 *Figure 2. Predicted last 100  labels for testset2, for all four bearings*

 ![Figure 3](./images/LogisticRegression/testset2_result.jpg)  
 *Figure 3. Result for the bearcing which is failed for the testset2*


**For Kmeans Clustering**:

Fit the result data frame into Kmeans cluster, grouping into 8 cluster(number is obtained from elbow method). Save the trained model into Kmeans.

For testing, take input path from the user, to check which bearing is suspected to fail and which is in normal condition. Calculate the number of label 5, 6, 7 for the last 100 predictions. If this is greater than 25, then it is suspected to fail. Otherwise, this is in normal condition.

- Copy the dataset to Kmeans/<br />
    cd path_to_Kmeans<br />
    cp -r path_to_1st_test .<br />
    cp -r path_to_2nd_test .<br />

- Train Kmeans<br />
    python3 kmeanstrainingall.py<br />
    <br />
    enter the complete directory path for the testset1 <I>path_to_Kmeans/1st_test/</I><br />
    enter the complete directory path for the testset2 <I>path_to_Kmeans/2nd_test/</I><br />

- Test the model<br />
    python3 kmeanstestingall.py <br />
    <br />
    enter the complete directory path <I>path_to_Kmeans/1st_test/</I><br />
    enter the folder name <I>1st_test/</I><br />

![Figure 4](./images/Kmeans/testset2_figure.jpg)  
 *Figure 4. Predicted last 100 labels for testset2, for all four bearings using the Kmeans clustering*

 ![Figure 5](./images/Kmeans/testset2_result.jpg)  
 *Figure 5. Result of testset2 which bearing has failed using the Kmeans clustering*

**For GMM:**

Fit the result data frame into gmm model, grouping into 3 components (components depicts no of clusters). Save the trained model as GMM using numpy.

For the testing, take the input path from the user, to check which bearing is suspected to fail and which is in normal condition. Calculate the number of label 2 for the last 100 predictions. If this is greater than 50 then it is suspected to fail. Otherwise this is in normal condition.

- Copy the dataset to GMM/<br />
    cd path_to_GMM<br />
    cp -r path_to_1st_test .<br />
    cp -r path_to_2nd_test .<br />

- Train GMM<br />
    python3 GMM_train.py <br />
    <br />
    enter the complete directory path for the testset1 <I>path_to_GMM/1st_test/</I><br />
    enter the complete directory path for the testset2 <I>path_to_GMM/2nd_test/</I><br />

- Test the model:<br />
    python3 GMM_test.py <br />
    <br />
    enter the complete directory path <I>path_to_GMM/1st_test/</I><br />
    enter the folder name <I>1st_test/</I><br />

![Figure 6](./images/GMM/testset2_figure.jpg)  
 *Figure 6. Predicted last 100 labels for testset2, for all four bearings using the GMM clustering*


![Figure 7](./images/GMM/testset2_result.jpg)  
*Figure 7. Result of testset2 which bearing has failed using the GMM clustering*

**NOTE:**

TESTSET 3 of the NASA bearing dataset is discarded for the observations because of the following reason:

1: It has a 6324 data file in actuality, but according to the documentation it contains a 4448 data file. This makes very noisy data.

2: None of the bearing shows the symptom of failure. However, it suddenly fails. This makes the data inconsistent

These listed reasons are why testset3 shows unpredictable behavior.

## (Optional) Saving Data to the Cloud
As an optional step you can take the data results and send these to an Amazon Web Services (AWS)*  instance for graphing.

1\. Make an EC2 Linux* instance on AWS
([*https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2\_GetStarted.html*](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html))

2\. Install InfluxDB* on EC2 Linux instance
([*https://github.com/influxdata/influxdb*](https://github.com/influxdata/influxdb))

3\. Install Grafana on EC2 Linux instance
([*https://grafana.com/get*](https://grafana.com/get))
