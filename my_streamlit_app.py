from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import streamlit as st

# Load the sentiment analysis model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis")

# Create a Streamlit web app
st.title("Sentiment Analysis of Customer Reviews")

# Allow the user to upload a review file
uploaded_file = st.file_uploader("Upload a review file", type=["txt"])

# Define sentiment labels
sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

if uploaded_file is not None:
    # Read the uploaded file
    text = uploaded_file.read().decode("utf-8")

    # Tokenize and classify sentiment
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax().item()

    # Determine the sentiment
    sentiment = sentiment_labels.get(predicted_class, "Unknown")

    # Display the sentiment and the original text
    st.subheader("Sentiment Analysis Result")
    st.write(f"Sentiment: {sentiment}")
    st.subheader("Original Text")
    st.write(text)