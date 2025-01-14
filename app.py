import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_file
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import sklearn
from werkzeug.utils import secure_filename
from flask import abort
import joblib
from joblib import load, dump
import pandas as pd
import string
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import sklearn
print(f"Current scikit-learn version: {sklearn.__version__}")
app = Flask(__name__)
#load our trained model
random_forest_model = load('random_forest_model.joblib')

matplotlib.use('Agg')

def generate_pie(df):
    counts = df['Prediction'].value_counts()
    labels = counts.index.map({1: 'Yes', 0: 'No'})
    # Create a 3D pie chart
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['blue', 'red'])
    plt.title('Distribution of Predictions')
    # Save plot 
    plot_image_path = 'static/pie_plot.png'
    plt.savefig(plot_image_path)
    plt.clf()

def generate_plots(df):

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Plot for each column
    for i, col in enumerate(['education', 'housing', 'contact']):
        grouped_df = df.groupby(col)['Prediction'].value_counts().unstack().fillna(0)
        grouped_df.plot(kind='bar', stacked=True, ax=axes[i])
        axes[i].set_title(f'Count of {col} categories for each Prediction')
        axes[i].set_xlabel(f'{col}')
        axes[i].set_ylabel('Count')
        axes[i].legend(title='Predictions', labels=['No', 'Yes'])

    plt.tight_layout()
    # Save plot 
    plot_image_path = 'static/bar_plot.png'
    plt.savefig(plot_image_path)
    plt.clf()

    return plot_image_path

def generate_bar_plot(df, categorical_column, predicted_output):
    # Count of occurrences of each category in the specified column
    category_counts = df[categorical_column].value_counts()

    # Plot  bar graph
    plt.bar(category_counts.index, category_counts.values)

    # Customizing the plot
    plt.title(f'Bar Plot for {categorical_column}')
    plt.xlabel(categorical_column)
    plt.ylabel('Count')

    # Save plot 
    plot_image_path = 'static/bar_plot.png'
    plt.savefig(plot_image_path)

    # Clear current plot to avoid issues
    plt.clf()

    return plot_image_path
#creating API's and endpoints
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analysis')
def page1():
    return render_template('analysis.html')

# Creating a route to handle form submission and file upload
@app.route('/predict-file', methods=['POST'])
def predict_file():
    # Check if a file is included in the request
    if 'file' in request.files:
        file = request.files['file']

        # Check if the file has an allowed extension
        if file.filename != '' and file.filename.endswith('.csv'):
            # Read the CSV file
            df = pd.read_csv(file)
            df1 = df.copy()

            # Apply preprocessing to the entire DataFrame
            input_data = preprocess_input(df)

            # Make predictions using the loaded model
            predictions = random_forest_model.predict(input_data)
            prediction_prob = random_forest_model.predict_proba(input_data)
        
            # Add predictions to the DataFrame
            df1['Prediction'] = predictions

            df1['Chance of Enrolling'] = (prediction_prob[:, 1]*100).round(2).astype(str)
            df1['Chance of Enrolling'] = df1['Chance of Enrolling'] + '%'
            #image_path = generate_bar_plot(df1, 'education', 'Prediction')
            image_path = generate_plots(df1)
            generate_pie(df1)           
            # Convert the table to a string
            df1['Prediction'] = df1['Prediction'].replace({0: 'No', 1: 'Yes'})
            result_str = df1.to_html(index=False)

            return jsonify({'result': result_str, 'plot_image': image_path})  # Example percentage value
# Creating a route to handle form submission and prediction for a sinle user datset
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    education = request.form['education']
    housing = request.form['housing']
    contact = request.form['contact']
    duration = float(request.form['duration'])
    campaign = float(request.form['campaign'])
    pdays = float(request.form['pdays'])
    previous = float(request.form['previous'])
    poutcome = request.form['poutcome']

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'education': [education],
        'housing': [housing],
        'contact': [contact],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome]
    })

    # Call preprocess_input with the DataFrame
    processed_data = preprocess_input(input_data)

    # Make predictions using the loaded model
    prediction = random_forest_model.predict(processed_data)
    prediction_prob = random_forest_model.predict_proba(processed_data)
    if prediction == 0:
        result = f"The customer is unlikely to enroll in the Term-Deposit. Chance of enrolling is {prediction_prob[0][1]*100}%"
    elif prediction == 1:
        result = f"The customer is likely to enroll in the Term-Deposit. Chance of enrolling is {prediction_prob[0][1]*100}%"
    
    return jsonify({'result': result, 'percentage': prediction_prob[0][1]*100})


def preprocess_input(df):
    # Converting categorical variables to numerical, removing spaces string opeations etc.
    cat_columns = df.select_dtypes(include=['object']).columns.tolist()
    for c in cat_columns:
        df[c] = df[c].str.lower()

    # Removing special characters
    for c in cat_columns:
        df[c] = df[c].str.replace('[{}]'.format(string.punctuation), '')

    #  Remove any unnecessary spaces
    for c in cat_columns:
        df[c] = df[c].str.replace(' ', '')

    education_labels = {'primary': 0, 'secondary': 1, 'tertiary': 2}
    df['education'] = df['education'].map(education_labels)

    yn = {'no': 0, 'yes': 1}
    df['housing'] = df['housing'].map(yn)

    contact_labels = {'cellular': 0, 'telephone': 1, 'unknown': 2}
    df['contact'] = df['contact'].map(contact_labels)

    poutcome_labels = {'failure': 0, 'success': 1, 'unknown': 2, 'other': 3}
    df['poutcome'] = df['poutcome'].map(poutcome_labels)
    df['duration']=(df['duration']-258.32) / 258.16
    #  convert to list for prediction
    processed_data_list = df.values.tolist()

    return processed_data_list

app.debug = True

if __name__ == '__main__':
    app.run(debug=True)
