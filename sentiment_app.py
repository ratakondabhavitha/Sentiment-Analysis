import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
from indicnlp.tokenize import indic_tokenize
from sklearn.exceptions import NotFittedError

# Function to clean and tokenize code-mixed text
def clean_code_mixed_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        text = text.lower()  # Convert to lowercase
        tokens = indic_tokenize.trivial_tokenize(text)
        cleaned_text = " ".join(tokens)
        return cleaned_text
    return ''

# Hybrid model for combining TF-IDF + ML and BERT predictions
class HybridModel:
    def __init__(self, tfidf_model, ml_model, bert_pipeline):
        self.tfidf_model = tfidf_model
        self.ml_model = ml_model
        self.bert_pipeline = bert_pipeline
        self.is_fitted = False  # Track whether the model is fitted

    def fit(self, X_train, y_train):
        # Train TF-IDF + ML model
        X_train_tfidf = self.tfidf_model.fit_transform(X_train)
        self.ml_model.fit(X_train_tfidf, y_train)
        self.is_fitted = True  # Mark as fitted

    def predict(self, X):
        if not self.is_fitted:
            raise NotFittedError("The TF-IDF vectorizer is not fitted. Please train the model before making predictions.")
        
        # Predict using TF-IDF + Random Forest
        X_tfidf = self.tfidf_model.transform(X)
        ml_predictions = self.ml_model.predict(X_tfidf)

        # Predict using BERT
        bert_predictions = [1 if self.bert_pipeline(text)[0]['label'] == 'LABEL_1' else 0 for text in X]

        # Combine predictions (majority voting)
        final_predictions = [
            1 if (ml_pred + bert_pred) > 1 else 0
            for ml_pred, bert_pred in zip(ml_predictions, bert_predictions)
        ]
        return final_predictions

# Streamlit UI
st.title("Sentiment Analysis App - Hybrid Approach")

# Upload Dataset
uploaded_file = st.file_uploader("Upload a CSV file with 'sentence' and 'label' columns", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Check for missing values in the dataset
    if df.isnull().sum().sum() > 0:
        st.warning("The dataset contains missing values. They will be dropped before processing.")
        df = df.dropna()

    # Preprocess the text
    df['sentence'] = df['sentence'].apply(clean_code_mixed_text)

    # Handle invalid or missing labels
    valid_labels = ['positive', 'negative']
    df['label'] = df['label'].apply(lambda x: x if x in valid_labels else 'unknown')
    df['label'] = df['label'].map({'positive': 1, 'negative': 0, 'unknown': -1})

    # Check for invalid labels after mapping
    if (df['label'] == -1).sum() > 0:
        st.warning(f"There are {df['label'].sum()} rows with unknown labels, which will be excluded.")
        df = df[df['label'] != -1]

    # Split data
    X = df['sentence']
    y = df['label']

    # Final check for NaN in y
    if y.isnull().any():
        st.error("The dataset contains invalid or missing labels after preprocessing. Please check the input file.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    tfidf_vectorizer = TfidfVectorizer()
    rf_model = RandomForestClassifier()
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
    bert_pipeline = pipeline("sentiment-analysis", model=bert_model, tokenizer=bert_tokenizer)

    # Initialize hybrid model
    hybrid_model = HybridModel(tfidf_vectorizer, rf_model, bert_pipeline)

    if st.button("Train and Evaluate"):
        # Train the hybrid model
        hybrid_model.fit(X_train, y_train)
        st.session_state["hybrid_model"] = hybrid_model  # Save trained model in session state
        st.write("Model trained successfully!")

        # Evaluate the model
        predictions = hybrid_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)

        # Display results
        st.write(f"### Accuracy: {accuracy:.2f}")
        st.write("### Classification Report")
        st.json(report)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, predictions)
        st.write("### Confusion Matrix")
        st.write(conf_matrix)

    # Test User Input
    st.write("### Test Your Own Input")
    user_input = st.text_input("Enter a sentence to analyze:", "")
    if st.button("Analyze Input"):
        if "hybrid_model" in st.session_state and st.session_state["hybrid_model"].is_fitted:
            hybrid_model = st.session_state["hybrid_model"]
            cleaned_input = clean_code_mixed_text(user_input)
            prediction = hybrid_model.predict([cleaned_input])
            sentiment = "Positive" if prediction[0] == 1 else "Negative"
            st.write(f"Prediction: {sentiment}")
        else:
            st.write("Please train the model first!")
else:
    st.write("Please upload a CSV file to proceed.")
