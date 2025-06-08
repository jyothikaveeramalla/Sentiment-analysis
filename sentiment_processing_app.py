import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import plotly.express as px
import plotly.graph_objects as go

nltk.download('punkt')
nltk.download('stopwords')

@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

labels = ['Negative', 'Neutral', 'Positive']

def preprocess_text(text):
    tokens = word_tokenize(text)
    negation_words = {"not", "no", "never"}
    stop_words = set(stopwords.words('english')) - negation_words
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    removed_stopwords = [word for word in tokens if word.lower() in stop_words and word.lower() not in negation_words]
    return " ".join(filtered_tokens), removed_stopwords

def predict_sentiment(text):
    processed_text, removed_stopwords = preprocess_text(text)
    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()
    return labels[pred_class], confidence, removed_stopwords, probs[0].tolist()

def create_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': confidence * 100}}))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=10))
    return fig

def main():
    st.title("📚 Customer Review Sentiment Classifier")
    st.write("This app enhances input data using **tokenization** and **stop-word removal** before performing sentiment classification.")

    text_input = st.text_area("Enter a Customer Review:", height=70)

    if st.button("Analyze Review"):
        if text_input.strip():
            sentiment, confidence, removed_stopwords, prob_dist = predict_sentiment(text_input)

            # Create layout with columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Analysis Result:")
                st.write(f"**Predicted Sentiment:** {sentiment}")
                
                # Gauge chart for confidence
                st.plotly_chart(create_gauge(confidence), use_container_width=True)
                
                # Sentiment probability distribution
                st.write("**Sentiment Probability Distribution:**")
                fig = px.pie(values=prob_dist, names=labels, 
                            color=labels, 
                            color_discrete_map={'Negative':'red', 'Neutral':'gray', 'Positive':'green'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Sentiment bar chart
                st.write("**Sentiment Scores:**")
                fig = px.bar(x=labels, y=prob_dist, 
                            color=labels,
                            color_discrete_map={'Negative':'red', 'Neutral':'gray', 'Positive':'green'},
                            labels={'x': 'Sentiment', 'y': 'Probability'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Display removed stopwords
                st.subheader("Removed Stop Words:")
                if removed_stopwords:
                    st.write("**" + "**, **".join(sorted(set(removed_stopwords))) + "**")
                else:
                    st.write("No stop words were removed.")

        else:
            st.warning("Please enter a customer review to analyze.")

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit and Hugging Face Transformers.")

if __name__ == "__main__":
    main()