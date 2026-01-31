# Wine Quality Classification Project

A machine learning project that uses neural networks to classify wines as red or white based on their physicochemical properties.

## Overview

This project builds a binary classification model using Keras/TensorFlow to predict whether a wine is red or white based on 11 physicochemical features. The model achieves approximately 95% accuracy on the test set.

## Dataset

The project uses the Wine Quality dataset from the UCI Machine Learning Repository, containing:

- **Red Wine**: 1,599 samples
- **White Wine**: 4,898 samples
- **Total**: 6,497 wine samples

### Features

Each wine sample includes the following physicochemical properties:

1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol

### Target Variable

- **Type**: Binary classification (0 = White wine, 1 = Red wine)

## Project Structure

```
wine-quality-classification/
│
├── redwinequality.csv          # Red wine dataset
├── whitewinequality.csv        # White wine dataset
├── WIne_Testing.ipynb          # Main Jupyter notebook
└── README.md                   # Project documentation
```

## Requirements

```python
pandas
numpy
matplotlib
scikit-learn
keras
tensorflow
```

## Installation

1. Clone this repository or download the project files

2. Install required packages:
```bash
pip install pandas numpy matplotlib scikit-learn keras tensorflow
```

3. Ensure you have the dataset files in the same directory as the notebook

## Usage

### Running the Notebook

1. Open `WIne_Testing.ipynb` in Jupyter Notebook or Google Colab

2. Run all cells sequentially to:
   - Load and combine the datasets
   - Perform exploratory data analysis
   - Train the neural network model
   - Make predictions on test data

### Code Structure

The notebook follows these steps:

1. **Data Loading**
   - Loads red and white wine datasets
   - Adds type labels (1 for red, 0 for white)
   - Combines datasets and removes missing values

2. **Exploratory Data Analysis**
   - Visualizes alcohol content distribution by wine type
   - Displays side-by-side histograms for comparison

3. **Data Preprocessing**
   - Separates features (X) and target variable (y)
   - Splits data into training (80%) and test (20%) sets

4. **Model Building**
   - Creates a Sequential neural network with:
     - Input layer: 12 features
     - Hidden layer: 12 neurons with ReLU activation
     - Output layer: 1 neuron with sigmoid activation
   - Compiles with Adam optimizer and binary crossentropy loss

5. **Training**
   - Trains for 3 epochs with batch size of 1
   - Achieves ~95% accuracy

6. **Prediction**
   - Makes predictions on test data
   - Converts probabilities to binary labels (threshold: 0.5)

## Model Architecture

```
Sequential Model:
├── Dense Layer (12 units, relu activation)
├── Dense Layer (8 units, relu activation)
└── Dense Layer (1 unit, sigmoid activation)

Optimizer: Adam
Loss Function: Binary Crossentropy
Metric: Accuracy
```

## Results

- **Training Accuracy**: ~95.5% (after 3 epochs)
- **Model Performance**: Successfully distinguishes between red and white wines based on chemical properties

### Sample Predictions

The model can predict wine type from chemical properties:
```
Prediction: White wine
Prediction: White wine
Prediction: Red wine
...
```

## Key Insights

- The model effectively learns to distinguish between red and white wines using only physicochemical properties
- Alcohol content distribution differs between red and white wines, as shown in the exploratory analysis
- The neural network approach provides high accuracy with a simple architecture

## Future Improvements

Potential enhancements to consider:

- Add evaluation metrics (precision, recall, F1-score, confusion matrix)
- Implement cross-validation for more robust performance estimation
- Experiment with different network architectures
- Add regularization techniques (dropout, L2 regularization)
- Tune hyperparameters (learning rate, batch size, epochs)
- Include feature importance analysis
- Build a multi-class classifier to predict wine quality scores
- Deploy the model as a web application

## Data Source

Dataset: [UCI Machine Learning Repository - Wine Quality Dataset](http://archive.ics.uci.edu/ml/datasets/Wine+Quality)

**Citation**: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

## License

This project uses publicly available data from the UCI Machine Learning Repository. Please refer to the original dataset's license and citation requirements.

## Contributing

Feel free to fork this project and submit pull requests with improvements or open issues for bugs and feature requests.

## Contact

For questions or feedback about this project, please open an issue in the repository.

---

**Note**: This is an educational project demonstrating binary classification with neural networks using wine quality data.
