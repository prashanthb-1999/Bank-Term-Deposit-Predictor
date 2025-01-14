# Bank Term Deposit Predictor

## Project Overview

This project aims to develop a predictive model to determine whether a customer will subscribe to a term deposit based on direct marketing phone calls made by a Portuguese banking institution. The goal is to optimize the bank's marketing strategy by identifying potential customers for term deposits.

## Features

- **Predictive Model**: Utilizes machine learning to predict customer subscription to term deposits.
- **Web UI Tool**: Allows bank employees to input customer data and get predictions.
- **Single Customer Prediction**: Input individual customer data for prediction.
- **Batch Prediction**: Upload a CSV file containing multiple customer records for batch predictions.
- **Analysis Charts**: Visual representation of customer trends and behaviors.
- **Recommendations**: Provides targeted customer recommendations based on predictions.

## Dataset

The dataset used for training the model is available on [Kaggle](https://www.kaggle.com/datasets/hariharanpavan/bank-marketing-dataset-analysis-classification).

## Folder Structure
  - `templates/`: HTML files for the web UI.
  - `static/`: Static files such as generated plots.
  - `test-dataset1.csv` & `test-dataset2.csv`: Sample test datasets.
  - `random_forest_model.joblib`: Pre-trained Random Forest model.
  - `app.py`: Main Flask application.

## Instructions

### Running the Application

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/bank-term-deposit-predictor.git
2.Instructions:
1. Run the app.py file to start the main application. You can do this either from a code editor like Visual Studio or from the terminal.
a. For Visual Studio, Click on Run without Debugging
b. For terminal, Run the command: python app.py
Note: If there is an error related to loading model joblib, install the correct sklearn version which is 1.2.2. (In the terminal, Run pip install scikit-learn==1.2.2 ).The version of scikit-learn used while saving the model is 1.2.2 .Hence it is important to use same version to load the model without any errors.
2. When the app has started, you can see the logs describing on which address/port the application is running on as seen below. Go to this address on any web browser and you will be able to see the home page of the application
Note: The application generally runs on http://127.0.0.1:5000/ but is not always the case.
  3. On the Home Page, if you click on the ‘Enter Customer Details’ button, a form will be displayed where Customer details can be entered and when you click on ‘Predict’, the output will show how likely the customer is to enroll in a term-deposit.
Sample Input:
• Education: primary
• Housing: yes
• Contact: cellular
• Duration: 1000
• Campaign: 4
• Pdays: 200
• Previous: 4
• Poutcome: success
  ![Project Screenshot](https://github.com/prasanthmanda/Bank-Term-Deposit-Predictor-Application/blob/main/Picture1.png)      
 4. If you click on the ‘Enter Customer Dataset’ button on top of the page, you will see an option to upload a CSV file. You can test this functionality by running the application and upload the sample test datasets (test-dataset1.csv, test-dataset2.csv) present in the src/phase3 folder. The output will show the predictions along with the probability of enrolling in a tabular format as seen below:
  ![Project Screenshot](https://github.com/prasanthmanda/Bank-Term-Deposit-Predictor-Application/blob/main/Picture2.png)   
5. If you click on the ‘Analysis Charts’ button on top of the page, you will see the generated plots/charts related to the uploaded customer dataset depicting various patterns and trends of the customers. This can be seen below:
  ![Project Screenshot](https://github.com/prasanthmanda/Bank-Term-Deposit-Predictor-Application/blob/main/Picture3.png)
6. If you click on the ‘User Guide’ button on top of the page, you will see instructions on how to use the data product like this instruction guide.
![Project Screenshot](https://github.com/prasanthmanda/Bank-Term-Deposit-Predictor-Application/blob/main/Picture4.png)
