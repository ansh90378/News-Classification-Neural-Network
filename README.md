# Fake News Detection Neural Network

## Overview
This project implements a **Neural Network** for detecting whether a news article is **fake** or **true**. The model is designed for binary classification and was trained using Python and PyTorch. It can be deployed as a web application using **Streamlit**.

## Features
- **Neural Network Architecture**: A simple yet effective neural network to classify news articles.
- **Deployment with Streamlit**: An interactive web application for users to test the model.
- **Pre-trained Weights**: Best weights of the trained model included for quick deployment.
- **Text Vectorizer**: A vectorizer is provided to preprocess input text data before feeding it into the model.

## Files Included
1. **notebook.ipynb** - The Jupyter notebook containing the code for data preprocessing, training, and evaluation.
2. **app.py** - Streamlit application code to deploy the model.
3. **model_weights.pth** - Pre-trained weights for the neural network.
4. **vectorizer.pkl** - Pre-trained vectorizer for text processing.

## Dataset
- The dataset used for training the model can be accessed [here](<insert-dataset-link>).

## Requirements
- Python 3.8 or later
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Matplotlib
- Seaborn

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```bash
   cd fake-news-classifier
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Training the Model
- Open the Jupyter notebook:
  ```bash
  jupyter notebook notebook.ipynb
  ```
- Follow the steps in the notebook to preprocess the data, train the model, and save the best weights.

### 2. Running the Web Application
- Start the Streamlit app:
  ```bash
  streamlit run app.py
  ```
- Open the URL displayed in the terminal to interact with the application.

### 3. Testing the Model
- Input the title and text of a news article.
- The model will output whether the article is **fake** or **true** based on the input.

## Model Details
- **Input Features**: Processed text data using vectorization.
- **Architecture**: Multi-layer feedforward neural network with ReLU activations and Sigmoid output.
- **Output**: Probability score (0 to 1) indicating whether the article is **fake**.

## Model Performance
- **5-Fold Cross-Validation Results**:
  - Average Accuracy: **0.9735**
  - Average AUC: **0.9957**

## Example
```python
import torch
from model import FakeNewsBinaryClassifier

model = FakeNewsBinaryClassifier(input_dim=300)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Example input
sample_input = vectorizer.transform(["Sample news text..."]).toarray()
output = model(torch.tensor(sample_input, dtype=torch.float32))
print(output)
```

## Future Improvements
- Implement LSTM or Transformer-based models for improved performance.
- Add support for multi-class classification.
- Enhance preprocessing with NLP techniques like Word2Vec or BERT embeddings.
