# Machine_Learning_Optimization_Study
## 1. Project Overview

The project aims to apply machine learning techniques—Support Vector Machine (SVM) and Neural Networks (NN)—to analyze the provided dataset and optimize classifiers for performance. The project compares the accuracy and training time of different machine learning models.

### Key Focus:
- **Support Vector Machine (SVM)**: Trained using different kernel types (linear, polynomial, and RBF) to classify data and optimize accuracy and training time.
- **Neural Networks (NN)**: Optimized using various hyperparameters such as learning rate, batch size, number of layers, and epochs.

## 2. Files in this Repository

- `Learning From Data report.pdf`: The report containing the results and analysis of both SVM and Neural Networks.
- `Learning From Data code.ipynb`: The Jupyter notebook containing the code used to process the data, train the models, and generate visualizations.
- lfd_2023_group8.csv: The original dataset used for training and testing the models.

## 3. Instructions for Running the Code

To run the notebook (`Learning From Data code.ipynb`), ensure you have the following installed:
- **Python 3.13.0**
- Libraries: `pandas`, `matplotlib`, `seaborn`, `sklearn`, `tensorflow` (for the neural networks)

### Steps:
1. Install the necessary libraries using:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn tensorflow
   ```
2. Open the Jupyter notebook in your environment and run the cells sequentially to replicate the results.
3. The notebook includes:
   - Data preprocessing (handling missing and categorical values)
   - SVM model training with various kernels (linear, polynomial, RBF)
   - Hyperparameter tuning for both SVM and Neural Networks
   - Visualizations of model performance (accuracy and training time)

### Expected Outputs:
- Model performance metrics (accuracy and training time) for SVM and Neural Networks.
- Visual comparison of different kernel types in SVM and optimal hyperparameters for the neural network.

## 4. Analysis Overview

### 4.1 SVM Classifier
- **Data Preprocessing**: Missing values in categorical columns (e.g., "location") were imputed using the mode, and categorical data was encoded.
- **Model Training**: The SVM model was trained using three different kernels—linear, polynomial, and RBF. The linear kernel achieved the best balance between accuracy and training time.
- **Results**:
  - **Best model**: Linear SVM with C=0.1, achieving **98% accuracy** on the test set.
  - **Training time**: The linear kernel required significantly less time compared to polynomial and RBF kernels.

### 4.2 Neural Network Classifier
- **Hyperparameters Tuned**: Learning rate, batch size, number of layers, and epochs were adjusted to optimize performance.
- **Results**:
  - The neural network achieved **99.5% training accuracy** with the following parameters:
    - Learning rate: 0.005
    - Batch size: 256
    - Epochs: 100
    - Layers: 2 dense layers (48 and 24 neurons)
  - **Testing accuracy**: 97%, with more training time than SVM.

### 4.3 Model Comparison
- **SVM**: Faster training with comparable test accuracy (97-98%) across different kernels.
- **Neural Network**: Higher training accuracy but required significantly longer training time.

## 5. Conclusion

Both the SVM and neural network models achieved high accuracy, with SVM (linear kernel) providing a more efficient balance between accuracy and training time. The neural network demonstrated superior training accuracy but with longer computational costs.

## 6. References

A list of references used in the report, including key sources on machine learning, hyperparameter tuning, and optimization techniques.

## 7. Acknowledgements

This project was completed as part of the Learning From Data course, and the datasets were provided for educational purposes.

