import streamlit as st
import pandas as pd
from transformers import pipeline

# Load pre-trained text classification model (sentiment analysis)
classifier = pipeline('sentiment-analysis')

# Streamlit web application
def main():
    st.title("Text Classification App")

    # Option to either input text or upload a file
    option = st.radio("Choose input method:", ["Enter Text", "Upload File"])

    if option == "Enter Text":
        # Input text area
        text_input = st.text_area("Enter your text here:", "Type your text here...")
        if st.button("Get Classification"):
            # Make prediction
            result = classifier(text_input)

            # Display result
            st.write("Prediction:")
            st.write(f"Label: {result[0]['label']}")
            st.write(f"Score: {result[0]['score']:.4f}")
    else:
        # File upload
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            # Load the dataset
            df = pd.read_csv(uploaded_file)

            # Make predictions for each text in the dataset
            predictions = [classifier(text) for text in df['text']]

            # Display overall results
            st.write("Overall Results:")
            for i, prediction in enumerate(predictions):
                st.write(f"Row {i + 1}:")
                st.write(f"Label: {prediction[0]['label']}")
                st.write(f"Score: {prediction[0]['score']:.4f}")
                st.write("---")

if __name__ == "__main__":
    main()
