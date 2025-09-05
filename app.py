# app.py - News Topic Classifier Web App

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Page config
st.set_page_config(
    page_title="📰 News Topic Classifier",
    page_icon="📰",
    layout="centered"
)

# App title
st.title("📰 News Topic Classifier")
st.markdown("Classify news headlines into: World, Sports, Business, Sci/Tech")
st.write("")

# Load model and tokenizer (cached to speed up reload)
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-news-classifier-finetuned")
        model = AutoModelForSequenceClassification.from_pretrained("bert-news-classifier-finetuned")
        return tokenizer, model
    except Exception as e:
        st.error("❌ Error loading model. Make sure:")
        st.error("1. Folder `bert-news-classifier-finetuned` exists")
        st.error("2. It contains `pytorch_model.bin`, `config.json`, etc.")
        st.stop()

tokenizer, model = load_model()

# Label mapping
label_names = ['World', 'Sports', 'Business', 'Sci/Tech']

# Input box
text = st.text_area(
    "Enter a news headline:",
    placeholder="Example: 'Tesla unveils new AI-powered robotaxi'"
)

# Predict button
if st.button("🔍 Classify"):
    if not text.strip():
        st.warning("Please enter a headline.")
    else:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        
        # Predict
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_id = logits.argmax().item()
        confidence = torch.softmax(logits, dim=1).max().item()

        # Show result
        st.success(f"**Predicted Topic:** `{label_names[pred_id]}`")
        st.info(f"**Confidence:** `{confidence:.2f}`")