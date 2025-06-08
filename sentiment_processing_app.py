import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import plotly.express as px
import plotly.graph_objects as go
import os
import ssl

# =============================================
# NLTK DATA DOWNLOAD FIX
# =============================================
try:
    # Create a custom SSL context to prevent download issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Set NLTK data path
    nltk_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    os.makedirs(nltk_dir, exist_ok=True)
    nltk.data.path.append(nltk_dir)

    # Download required data with explicit verification
    if not nltk.data.find("tokenizers/punkt"):
        nltk.download('punkt', download_dir=nltk_dir, quiet=True)
    if not nltk.data.find("corpora/stopwords"):
        nltk.download('stopwords', download_dir=nltk_dir, quiet=True)
except Exception as e:
    st.error(f"Error setting up NLTK: {str(e)}")
    st.stop()
# =============================================

@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Load model with error handling
try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

labels = ['Negative', 'Neutral', 'Positive']

def preprocess_text(text):
    try:
        tokens = word_tokenize(text)
        negation_words = {"not", "no", "never"}
        stop_words = set(stopwords.words('english')) - negation_words
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        removed_stopwords = [word for word in tokens if word.lower() in stop_words and word.lower() not in negation_words]
        return " ".join(filtered_tokens), removed_stopwords
    except Exception as e:
        st.error(f"Text processing error: {str(e)}")
        return text, []

# Rest of your existing functions (predict_sentiment, create_gauge) remain the same
# ...

def main():
    st.title("📚 Customer Review Sentiment Classifier")
    st.write("This app analyzes customer review sentiment using advanced NLP techniques.")

    text_input = st.text_area("Enter a Customer Review:", height=70)

    if st.button("Analyze Review"):
        if text_input.strip():
            try:
                sentiment, confidence, removed_stopwords, prob_dist = predict_sentiment(text_input)
                
                # Display results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Analysis Result:")
                    st.write(f"**Predicted Sentiment:** {sentiment}")
                    st.plotly_chart(create_gauge(confidence), use_container_width=True)
                    
                    st.write("**Sentiment Probability Distribution:**")
                    fig = px.pie(values=prob_dist, names=labels, 
                                color=labels, 
                                color_discrete_map={'Negative':'red', 'Neutral':'gray', 'Positive':'green'})
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.write("**Sentiment Scores:**")
                    fig = px.bar(x=labels, y=prob_dist, 
                                color=labels,
                                color_discrete_map={'Negative':'red', 'Neutral':'gray', 'Positive':'green'},
                                labels={'x': 'Sentiment', 'y': 'Probability'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Removed Stop Words:")
                    if removed_stopwords:
                        st.write("**" + "**, **".join(sorted(set(removed_stopwords))) + "**")
                    else:
                        st.write("No stop words were removed.")

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
        else:
            st.warning("Please enter a customer review to analyze.")

if __name__ == "__main__":
    main()
