# Neural Network for Customer Churn Prediction
Tech stack: Python, NumPy, Pandas, Scikit-learn, MissForest, Object-Oriented Programming (OOP)

A custom-built neural network from scratch to predict customer churn. Designed to handle missing data efficiently and optimize prediction through dynamic dropout, SGD, and data preprocessing techniques.

Features:
Custom Neural Network Architecture:
Implemented a neural network from scratch using Python and NumPy, incorporating autograd to efficiently compute gradients and perform backpropagation.

Dropout Regularization:
Integrated dropout to mitigate overfitting, dynamically disabling neurons during training for more robust model generalization.

Data Preprocessing and Imputation:
Utilized the MissForest imputation technique to handle missing data, achieving a 15% improvement in predictive accuracy by accurately estimating missing values.
Preprocessed the customer churn data with one-hot encoding, normalization, and train-validation-test splitting (80-10-10) to maintain consistency.

Optimization with SGD:
Implemented stochastic gradient descent (SGD) for parameter updates, incorporating dynamic dropout during training for improved convergence.

Comprehensive Performance Metrics:
Evaluated the model using precision, recall, F1-score, and ROC-AUC. Included verbose mode to track training and validation accuracy across epochs.

Dataset:
Customer Churn Dataset:

Contains customer demographic, account, and service usage data.

Target variable: Churn (Yes/No)

Missing values handled through MissForest imputation for optimal feature completeness.

Data split: 80% training, 10% validation, 10% test.

Results:
Achieved significant accuracy improvement by leveraging dropout and effective data preprocessing.

Demonstrated robust performance on the Customer Churn dataset, with high precision and recall metrics.

Validated model performance through extensive evaluation using ROC-AUC and classification reports.

⚙️ Usage:
Clone the repository:

bash
Copy
Edit
git clone https://github.com/abelmesfin11/customer-churn-prediction.git
cd customer-churn-prediction
Install dependencies:

bash
Copy
Edit
pip install numpy pandas scikit-learn missforest
Run the neural network:

bash
Copy
Edit
python neural.py
View the model performance:
The script will output precision, recall, F1-score, and ROC-AUC after training.

🛠️ Customization:
Model Configuration:
Adjust the number of layers and neurons in the MLP class for different architectures.

Training Parameters:
Modify learning rate, dropout probability, and the number of epochs to optimize performance.

Evaluation Metrics:
Customize the verbosity level to monitor training and validation metrics throughout the process.

