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
            avg_score = sum(prediction[0]['score'] for prediction in predictions) / len(predictions)
            st.write(f"Average Sentiment Score: {avg_score:.4f}")
            st.write("---")

    # Input text area for user
    user_input = st.text_area("Enter your text here:", "Type your text here...")

    # Button to get sentiment prediction for user input
    if st.button("Get Classification"):
        user_prediction = classifier(user_input)
        st.write("User Input Prediction:")
        st.write(f"Label: {user_prediction[0]['label']}")
        st.write(f"Score: {user_prediction[0]['score']:.4f}")

if __name__ == "__main__":
    main()
