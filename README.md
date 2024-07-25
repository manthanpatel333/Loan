Loan Default Prediction
This project aims to predict loan default probability using various machine learning models, including neural networks, decision trees, and support vector machines.

Project Overview
The goal of this project is to develop a model that predicts whether a borrower will default on a loan based on various features such as age, income, loan amount, credit score, months employed, number of credit lines, interest rate, loan term, and debt-to-income ratio (DTI).

Dataset
The dataset used in this project contains the following features:

Age
Income
Loan Amount
Credit Score
Months Employed
Number of Credit Lines
Interest Rate
Loan Term
Debt-to-Income Ratio (DTI)
Default (Target Variable)
Models Used
Logistic Regression: A simple and interpretable model for binary classification.
Decision Trees: Tree-based methods that are easy to interpret.
Random Forest: An ensemble method that improves performance by combining multiple decision trees.
Gradient Boosting Machines (GBMs): Advanced ensemble methods like XGBoost for better accuracy.
Support Vector Machines (SVM): Effective in high-dimensional spaces.
Neural Networks: Suitable for capturing complex relationships in the data.
Neural Network Model
A neural network was implemented with the following structure:

Input Layer: 9 features
Hidden Layers: Two hidden layers with 64 neurons each and ReLU activation
Output Layer: 1 neuron with sigmoid activation
Training
The model was trained using:

Optimizer: Adam
Loss Function: Binary Cross-Entropy
Epochs: 50
Batch Size: 8
Validation Split: 20%
Evaluation
The model's performance was evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

Results
The best-performing model achieved the following results:

Accuracy: 88.57%
Precision: 0.89 for non-default, 0.59 for default
Recall: 0.99 for non-default, 0.06 for default
Prediction Function
A function predict_default was created to predict the probability of default for new data points.

python
Copy code
def predict_default(age, income, loan_amount, credit_score, months_employed, num_credit_lines, interest_rate, loan_term, dti_ratio):
    new_data = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'LoanAmount': [loan_amount],
        'CreditScore': [credit_score],
        'MonthsEmployed': [months_employed],
        'NumCreditLines': [num_credit_lines],
        'InterestRate': [interest_rate],
        'LoanTerm': [loan_term],
        'DTIRatio': [dti_ratio],
    })
    
    new_data_scaled = scaler.transform(new_data)
    probability = model.predict(new_data_scaled)
    prediction = (probability > 0.5).astype("int32")
    
    return prediction[0][0], probability[0][0]
Usage
To use this function, simply provide the input features, and it will return the binary prediction (default or no default) and the probability of default.

python
Copy code
result, prob = predict_default(
    age=45, 
    income=55000, 
    loan_amount=120000, 
    credit_score=700, 
    months_employed=24, 
    num_credit_lines=5, 
    interest_rate=7.5, 
    loan_term=36, 
    dti_ratio=0.4,
)
print(f"Prediction: {'Default' if result == 1 else 'No Default'}, Probability: {prob:.2f}")
Conclusion
This project demonstrates the process of building and evaluating a machine learning model to predict loan default. The neural network model showed promising results, although there is room for improvement, especially in handling class imbalance.
