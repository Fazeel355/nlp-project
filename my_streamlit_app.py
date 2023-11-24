import streamlit as st
import pandas as pd
from transformers import pipeline

# Load pre-trained text classification model (sentiment analysis)
classifier = pipeline('sentiment-analysis')

# Streamlit web application
def main():
    st.title("Text Classification App")

    # File upload
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

    # Input text area
    user_text_input = st.text_area("Enter your text here:", "Type your text here...")

    if uploaded_file is not None:
        try:
            # Determine file type and load the dataset
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                st.stop()
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty.")
            st.stop()
        except pd.errors.ParserError as e:
            st.error(f"Error parsing the file: {e}")
            st.stop()

        # Check if 'text' column exists
        if 'text' not in df.columns:
            st.error("The 'text' column is missing in the dataset.")
        else:
            # Make predictions for each text in the dataset
            predictions = [classifier(text) for text in df['text']]

            # Display overall results
            st.write("Overall Results:")
            for i, prediction in enumerate(predictions):
                st.write(f"Row {i + 1}:")
                st.write(f"Label: {prediction[0]['label']}")
                st.write(f"Score: {prediction[0]['score']:.4f}")
                st.write("---")

            # Calculate average sentiment scores
            avg_positive_score = sum(prediction[0]['score'] for prediction in predictions if prediction[0]['label'] == 'POSITIVE') / len(predictions)
            avg_negative_score = sum(prediction[0]['score'] for prediction in predictions if prediction[0]['label'] == 'NEGATIVE') / len(predictions)

            # Display average sentiment scores
            st.write("Average Sentiment Scores:")
            st.write(f"Positive: {avg_positive_score:.4f}")
            st.write(f"Negative: {avg_negative_score:.4f}")

    # Perform sentiment analysis for user-entered text
    if st.button("Analyze User Text"):
        user_result = classifier(user_text_input)
        st.write("User Input Analysis:")
        st.write(f"Label: {user_result[0]['label']}")
        st.write(f"Score: {user_result[0]['score']:.4f}")

if __name__ == "__main__":
    main()
