Step-by-Step Guide to Build a CNN for Multi-Label Classification Problems:
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
