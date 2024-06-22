# Political Sentiment Analysis

This repository contains the code and data for analyzing political sentiment in tweets using a neural network model. The goal of the project is to classify tweets into two predominant political affiliations: Democrat or Republican, based on their content.

## Project Overview

News sources can exhibit political bias due to various factors, including ownership, editorial stance, and journalists' perspectives. Recognizing this bias is crucial for media literacy and critical consumption of information. This project uses a sentiment analysis model to identify bias within news sources by analyzing the underlying sentiment expressed in the language used.

The sentiment analysis model is implemented using a feed-forward neural network and applied to a dataset of tweets made by Republican or Democratic political figures. The model achieved an accuracy of 75% on the test set.

## Files in the Repository

- `Political_Analysis.ipynb`: The Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- `Project-Report.pdf`: The detailed project report, including methodology, experiments, and results.

## Dataset

The dataset used includes tweets made by political figures in the United States. The data has labels for Republican or Democrat, the username of the tweeter, and the text of the tweet itself. There are 433 unique users that generated almost 85,000 tweets. The dataset is split into training and test sets with an 80-20 split.

## Methodology

The project's cornerstone is a deep learning model designed to classify political bias in text. The neural network architecture includes:

1. **Embedding Layer**: Converts words into dense vectors.
2. **Flatten Layer**: Reshapes the output into a one-dimensional array.
3. **Batch Normalization Layer**: Normalizes inputs to maintain stable activations.
4. **Dropout Layer**: Prevents overfitting by randomly deactivating neurons during training.
5. **Dense Layer**: Fully connected layer with ReLU activation and L2 regularization.
6. **Output Layer**: Uses softmax activation to produce a probability distribution.

The model is trained using the Adam optimizer and categorical cross-entropy loss function. Training is conducted over 10 epochs with a batch size of 64, and early stopping and learning rate reduction are employed to prevent overfitting.

## How to Use

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/Political-Sentiment-Analysis.git
    cd Political-Sentiment-Analysis
    ```

2. **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook Political_Analysis.ipynb
    ```

## Results

The model was evaluated on a test set of 17,000 tweets, achieving an accuracy of approximately 75%. Error analysis indicated that the model initially struggled with non-content parts of the tweets, which was mitigated by cleaning the dataset.

## Conclusion

The model demonstrates a commendable accuracy rate of 75% in distinguishing whether a given political tweet originated from a Republican or Democrat. However, the dataset's narrow scope may limit the model's applicability to broader political texts. Further exploration and refinement are needed to enhance its generalizability.

## In-Depth Report

For a detailed explanation of the project, including related works, data preprocessing, model architecture, and experimental results, please refer to the 
[Project Report][Project-Report.pdf](https://github.com/user-attachments/files/15934757/Project-Report.pdf)
