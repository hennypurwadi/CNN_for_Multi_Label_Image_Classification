## Building a CNN Model for Multi-Label Image Classification: A Comparative Study on Activation Functions and Loss Functions Using the Triple MNIST Dataset.

This is a collaborative Project with Nareshwar: https://github.com/nareshwar/uhull-module-2
Dataset link: https://huggingface.co/datasets/khushpatel2002/triple-mnist/tree/main

## Abstract

This project presents a detailed approach to develop a Convolutional Neural Network (CNN) for multi-label image classification using the Triple MNIST dataset, which contains images of three handwritten digits. We explore the impact of different activation functions such as SoftMax vs. Sigmoid, and comparing loss functions such as categorical cross-entropy vs. binary cross-entropy on model performance. The project follows a step-by-step methodology for data preprocessing, model building, and hyperparameter tuning, using techniques such as K-fold cross-validation, early stopping, and learning rate scheduling to optimize the model. Post-training evaluations using Grad-CAM and saliency maps provide the insights into the model's decision-making process. The study concludes with recommendations for applying these techniques to real-world multi-label classification tasks and suggests further research using this dataset.

## Research Question
How do different activation and loss functions (Sigmoid vs. SoftMax, categorical cross-entropy vs. binary cross-entropy) impact the performance of Convolutional Neural Networks (CNNs) in multi-label digit classification on the Triple MNIST dataset. What are the effects of hyperparameter tuning on the model’s accuracy and loss?

===================================================================================

## Step-by-Step Guide to Build a CNN for Multi-Label Classification Problems:

1. Import Libraries and Function
Begin by importing all the required libraries and functions for data processing, model building, and evaluation.

2. Load the Dataset
Load the training, validation, and test datasets from the specified directories.

3. Preprocess the Dataset
Reshape: Adjust the shape of the data to fit the model input requirements. In CNN need 4 dimensions of input shapes.
Normalize Pixel Values: Scale the pixel values to the range [0, 1].
Define Number of Categories: Specify the number of categories for each digit position.
Flatten: Convert multi-dimensional arrays to a single dimension.
Convert Labels to Categorical: Use one-hot encoding to convert labels for multi-label classification.
Flatten Label Lists: Convert the label lists to dictionaries for multi-output models.
Custom Data Generator: Define a custom data generator to yield batches of data during training.
4. Define Learning Rate and K-Fold Cross Validation
Set up a learning rate scheduler to adjust the learning rate during training.
Use K-fold cross-validation to better estimate model performance.
5. Set Up Callbacks for Early Stopping, and Model Checkpoint
Configure callbacks for early stopping, saving the best model, and logging training progress for visualization. Implement early stopping to prevent overfitting by stopping training when the validation loss stops improving.

6. Define Function to Plot Learning Curves
Create a function to visualize training and validation accuracy and loss over epochs.

7. Define the CNN Model Architecture
Set Up the Input Shape. Build the CNN model architecture. To be compared: The last layer/ Output layer uses a sigmoid versus softmax. The loss function uses categorical crossentropy versus binary crossentropy.

8. Train the Model
Train the model using the training data and validate it on the validation data, while using the defined callbacks.

9. Evaluate the Model
Evaluate the model and print the accuracy and loss metrics.

10. Perform Hyperparameter tuning
Perform 'keras-tuner' tuning to identify the best model parameters.
Apply regularization techniques to prevent overfitting.
Use cross-validation to obtain a more reliable estimate of the model's performance.
11. Re-Evaluate the Model
Re-Evaluate the model after Hyperparameter Tuning.

12. Final evaluation on the Test set
After finishing all the training and testing on the validation data, we trained the model one last time on all the training data. Then, we tested it on a completely new set of data that the model had never seen before (the Test set). This final evaluation checks how well the model performs on totally new data, making sure it’s ready for real life.

13. Visualization
After Final evaluation on the test set, we will visualize the results using a classification report, confusion matrix heatmaps, feature maps, Pydot, saliency map, Grad-CAM, Deep Dream Visualization, Guided Back Propagation Saliency map.

---------------------------------------------------------------------------

## Triple MNIST (Multi labels dataset) Project Report:

### Dataset Overview and Pre-processing 
Focusing on the objectives and importance of multi-label classification on multiple digits from images, where each image contains several digits. The dataset for this project consists of images each labelled with three digits.  

### Data Loading and Initial Exploration 
To load and explore the dataset, we define the load_images_from_folder function, which reads images from subdirectories, resizes them, converts them to grayscale, and assigns labels based on subdirectory names. Directories for training, validation, and test datasets are specified to organize the data. We print the contents of these directories to verify their structure. We load the datasets into arrays of images and corresponding labels. This process prepares the dataset for further processing and model training. 

### Pre-processing Steps 
Pre-processing steps were taken to prepare the data for modelling. 
This includes reshaping and normalizing pixel values, one-hot encoding of labels to convert labels to categorical, creation of a custom data generator for efficient data handling, setting up learning rate scheduler to adjust the learning rate during training, and configuring call-backs for early stopping to prevent overfitting by stopping the training when the validation loss stops improving. 

### Challenges 
The challenges during pre-processing involved reshaping the images to fit the model's input requirements, normalizing pixel values to the range [0, 1], and encoding the labels for a multi-label setup using one-hot encoding. We faced several challenges when we needed to choose which activation function and which loss function, we need to use. 

### Methodology and Techniques 
CNN models were developed, comparing categorical cross entropy with Binary cross entropy as loss functions and Sigmoid versus SoftMax as activation functions. Techniques like Early Stopping, Drop out were used to improve training performance and avoid overfitting. 

### Model Comparison and Selection 
Binary cross entropy as the loss function outperformed categorical cross entropy, showing lower loss and higher accuracy for multi-label classification, demonstrating its suitability for this task. While recommendations suggest Sigmoid for multi-label and SoftMax for multi-class, our experiment shows the use of SoftMax in the output layer achieved the highest accuracy. 

### Hyperparameter Tuning 
Hyperparameter tuning is performed using Keras-Tuner to find the best parameters, adjust the model parameters like learning rate and increase K-fold cross-validation to more accurately evaluate m del performance across different subsets of data, and implement batch normalization in the model to normalize the inputs for each mini-batch, reducing internal covariate shift and stabilizing training. 

### Results and Discussion: 
Initial models using Sigmoid activation and categorical cross entropy as the loss function achieved high accuracy, indicating no underfitting or overfitting. Switching to Sigmoid activation and Binary cross entropy as the loss function reduced losses (around 0.3), but lowered accuracy. The model using SoftMax activation and Binary Cross Entropy as the loss function showed better performance with higher accuracy and lower loss. 

Using Keras-Tuner, we found the best parameters and implemented into the model: 

val_loss: 0.07059618085622787 
Best val_loss So Far: 0.06173119693994522 
Total elapsed time: 05h 47m 46s  

'conv_1_filters': 96,  
'conv_2_filters': 128,  
'conv_3_filters': 128,  
'dense_units': 256 

After hyperparameter tuning, the model demonstrated improvement and better performance.  

digit_0_accuracy: 0.9817 
digit_1_accuracy: 0.9839 
digit_2_accuracy: 0.9847 
loss: 0.0987 

val_digit_0_accuracy: 0.9876 
val_digit_1_accuracy: 0.9858 
val_digit_2_accuracy: 0.9879 
val_loss: 0.0758 

In multi-class classification, the final layer of the model employs a SoftMax function to predict the class, and the training utilizes categorical cross-entropy as the loss function. Conversely, in multilabel classification, the last layer uses a sigmoid function to predict labels, with binary cross entropy serving as the loss function. (franky, 2018). But from our experiment, final testing on unseen data after Hyperparameter Tuning on SoftMax activation function and Binary Cross Entropy as the loss function achieved the model's highest accura
 cy (around 98%) and lowest loss (around 0.07).  
 
Test loss: 0.07689374685287476 
0.986299991607666 
0.9864500164985657 
0.9870499968528748 


#### Save the tuned model and load to predict on test image 
To know whether the trained model capable of making predictions on test images, after tuning and training the model, we save it to a file using the Keras format. To make predictions, we load the saved model and use it to make predictions on individual images from the test set. The model is designed to make accurate predictions on the test images. By loading the saved model and using it to predict the labels of a test image, we demonstrate its capability to correctly identify the labels. 

### Visualization 
We visualize the results using Classification Report, Confusion Matrix Heatmaps, Feature Maps, Saliency Maps, Grad-CAM, Deep Dream Visualization, and Guided Back Propagation Saliency maps. 

### Confusion Matrix 
The model shows excellent performance with high precision, recall, and F1-scores across all classes. The accuracy is 99%, indicating that the model correctly predicts most instances. The diagonal elements represent the correctly classified ones, with the highest values along the diagonal showing correct predictions. Off-diagonal values show where the model is making errors. Visual Representation of The Filters/ Kernels and Feature map Filter/Kernel is small matrices that slide over the input image to detect specific features such as edges, textures, or patterns. Each filter is applied across the entire input image, and the result is a feature map.  

The visual representation of the filters helps to understand what each filter is focusing on in the input images. The filters have dimensions (3, 3, 1, 96), where 3 and 3 are the height and width of each filter. 1 is the number of input channels for grayscale image; there is only one channel. 96 is the number of filters in this layer. The filters have purpose of detect specific features within the input images. During training, the weights of these filters are adjusted to minimize the loss function, thereby learning the features that are most relevant for the task. Visualizing the filters can give insights into what the model is learning. In this example, we chose a random handwritten image (number 430) from the training dataset and predicted the feature outputs.  The patterns seen in the feature show what each filter has learned. For example, some may detect edges, other textures, and some specific shapes. Early layers earn simple features, then deeper layers learn more complex features and patterns. 

Basic Saliency Map 
Basic Saliency maps highlight the most important areas of the input image at pixel level, showing which parts of the input image have the greatest impact on the model’s output prediction, regardless to make it better or worse. 

#### Grad-CAM 
Similar purpose with the Saliency Map but smoother, Grad-CAM highlights the most important areas of the input image at the region level, showing which parts of the input image contribute most to the output prediction. 

#### Deep Dream Visualization 
DeepDream visualization takes the input image(s), and enhances the patterns and curves what a network would typically see, and then create a dream-like visuals to showcase the learned features. 

#### Guided Back Propagation Saliency Map 
Unlike basic saliency map which allows positive and negative gradients if they make greatest contribution to the output prediction, guided back propagation saliency map method refined the basic saliency map by only allowing positive gradients to backpropagate through the network. Therefore, it refines the basic saliency map to highlight more relevant features. 

### Conclusion 
The project successfully demonstrated the application of CNNs to multi-label image-based digit classification, with SoftMax activation and binary cross entropy as the loss function showing the highest performance over Sigmoid activation and categorical cross entropy as the loss function. They are improving more after hyperparameter tuning. This is showing the importance of selecting appropriate loss functions and optimizing model architectures. Future work could explore OCR (Optical Character Recognition) Systems. The dataset can be utilized by researchers and developers to train and evaluate OCR systems for the recognition of handwritten multi-digit numbers. Furthermore, researchers could more complex architectures, additional regularization techniques, and further hyperparameter tuning to enhance model performance and robustness, potentially experimenting with different convolutional layer configurations to improve focus on relevant parts of the image. 

### Ethical, Legal, and Social Considerations in AI  
When developing AI models for multi-label image classification, such as handwritten triple MNIST digits, it's important to address ethical, legal, and social issues. Data protection laws like GDPR or CCPA must be adhered to, ensuring that any personal data is handled with strict privacy measures.  The ownership of the AI model, its training data, and its outputs need to carefully consider ensuring responsible and beneficial AI deployment. 
