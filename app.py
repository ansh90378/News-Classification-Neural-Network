import streamlit as st
import torch
import torch.nn as nn
import joblib

# Load the Model
class NewsClassifier(nn.Module):

    def __init__(self, input_dim):

        super(NewsClassifier, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 1),
                                    nn.Sigmoid())

    def forward(self, x):

        return self.network(x)

# load the model for inference
model = NewsClassifier(input_dim=1000)
model.load_state_dict(torch.load('news_classifier.pth', weights_only=True))

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app interface
st.title("Fake News Detection App")
st.write("Predict whether a news article is real or fake.")

# Input box
news_text = st.text_area("Enter the news article text:")

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text for prediction.")
    else:
        # Transform input text
        transformed_text = tfidf_vectorizer.transform([news_text])

        # Predict
        prediction = model.predict(transformed_text)
        prediction_prob = model.predict_proba(transformed_text)

        # Display results
        if prediction[0] == 0:
            st.success(f"The news is **REAL** with a confidence of {max(prediction_prob[0]) * 100:.2f}%.")
        else:
            st.error(f"The news is **FAKE** with a confidence of {max(prediction_prob[0]) * 100:.2f}%.")
