# Encryptix-2-Movie-Gerne-Classification
# Movie Genre Classification

This repository contains the implementation of a machine learning model to classify movie genres based on plot summaries. This project is part of the Encryptix Internship Task 2.

## Dataset

The dataset used for this project includes:
- `train_data.txt`: Training data containing movie plot summaries and corresponding genres.
- `test_data.txt`: Test data containing movie plot summaries for evaluation.
- `test_data_solution.txt`: Solutions for the test data.

## Methodology

1. **Data Preprocessing**
   - Clean and preprocess the text data (plot summaries).
   - Tokenization and vectorization using TF-IDF.

2. **Model Selection**
   - Experiment with various machine learning models:
     - Naive Bayes
     - Logistic Regression
     - Support Vector Machines

3. **Model Evaluation**
   - Evaluate the models based on accuracy and other relevant metrics.
   - Compare the performance of different models.

## How to Use

### Environment Setup


1. Clone the repository:

   ```sh
   git clone https://github.com/sammyatale/Encryptix-2-Movie-Gerne-Classification.git
   cd Encryptix-2-Movie-Gerne-Classification

Set up the Python environment:
pip install -r requirements.txt

### Running the Code
1. Train the model:     python code1.ipynb

2. Evaluate the model:  python app.py


### Files Description

train.py: Python script to train the machine learning model.
evaluate.py: Python script to evaluate the model on the test data.
code1.ipynb: Jupyter notebook with the main code implementation.
train_data.txt: Training data containing movie plot summaries and genres.
test_data.txt: Test data containing movie plot summaries.
test_data_solution.txt: Solutions for the test data.

### Results
The best performing model was Logistic Regression with an accuracy of 60%.
