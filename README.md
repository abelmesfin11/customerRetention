Neural Network for Customer Churn Prediction
This project implements a custom neural network from scratch to predict customer churn using the Customer Churn dataset. The neural network is built using Python and NumPy, incorporating advanced techniques such as autograd, dropout, and stochastic gradient descent (SGD). The model is evaluated using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

Features
Custom Neural Network: Built from scratch using Python and NumPy, implementing autograd for efficient gradient computation.

Data Preprocessing: Utilizes one-hot encoding, normalization, and missing data imputation using the MissForest algorithm.

Dropout Regularization: Integrates dropout layers to prevent overfitting and improve generalization.

Dynamic Training: Uses SGD for optimization, dynamically adjusting the model during training.

Performance Evaluation: Evaluates model accuracy, precision, recall, F1-score, and ROC-AUC.

Comprehensive Reporting: Uses classification_report from scikit-learn for detailed performance metrics.

Dataset
The model uses the Customer Churn dataset, which includes customer demographics, account information, and service usage. The target variable is whether a customer has churned (Yes or No).

Data Imputation: Missing values are imputed using the MissForest algorithm.

One-Hot Encoding: Categorical features are encoded to numerical format.

Normalization: Features are standardized for consistent model performance.

Model Architecture
The neural network consists of the following components:

Input Layer: Matches the number of features after preprocessing.

Hidden Layers: Three fully connected layers with ReLU activation and dropout regularization.

Output Layer: A single neuron with a sigmoid activation function for binary classification.

Loss Function: Negative Log-Likelihood Loss for effective probability estimation.

Optimizer: Stochastic Gradient Descent (SGD) for parameter updates.

Installation
Make sure you have Python and the necessary packages installed:

bash
Copy
Edit
pip install numpy pandas scikit-learn missforest
Usage
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
Run the script:

bash
Copy
Edit
python neural.py
Check the model performance:
The script will print accuracy, precision, recall, F1-score, and ROC-AUC after training.

File Structure
bash
Copy
Edit
customer-churn-prediction/
├── neural.py                # Main script containing the neural network implementation
├── CustomerChurn.csv        # Dataset file
├── README.md                # Project documentation
└── requirements.txt         # List of dependencies
Example Output
The output includes metrics such as:

markdown
Copy
Edit
Epoch 25: Training accuracy 85%, Validation accuracy 83%
              precision    recall  f1-score   support
           0       0.86      0.90      0.88       500
           1       0.82      0.77      0.79       300
    accuracy                           0.85       800
   macro avg       0.84      0.83      0.84       800
weighted avg       0.85      0.85      0.85       800
Customization
Model Configuration: Adjust the number of layers and neurons in the MLP class.

Learning Rate: Modify the learning rate and dropout probability as needed.

Epochs: Change the maximum number of training epochs for better convergence.

Acknowledgments

The MissForest algorithm was used for data imputation.

Scikit-learn was used for evaluation metrics.
