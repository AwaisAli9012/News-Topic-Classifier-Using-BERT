# 📰 News Topic Classifier Using BERT

A fine-tuned BERT model that classifies news headlines into 4 categories:
- 🌍 World
- ⚽ Sports
- 💼 Business
- 🔬 Sci/Tech

Built with **Hugging Face Transformers** and deployed using **Streamlit** for real-time interaction.

---

## 🚀 Overview

This project fine-tunes the `bert-base-uncased` model on the **AG News dataset** to perform text classification. After training, the model is deployed as an interactive web app where users can input headlines and get instant predictions.

Perfect for learning NLP, transfer learning, and lightweight model deployment.

---

## 🧩 Key Features

- ✅ Fine-tuned BERT for news classification
- ✅ Trained on the AG News dataset (Hugging Face)
- ✅ Evaluation using **Accuracy & F1-Score**
- ✅ Interactive web interface with **Streamlit**
- ✅ End-to-end pipeline: train → save → deploy

---

## 🖥️ How to Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/news-topic-classifier.git
cd news-topic-classifier
2. Install dependencies
pip install transformers torch datasets streamlit evaluate scikit-learn
3. Train the model
model.save_pretrained("bert-news-classifier-finetuned")
tokenizer.save_pretrained("bert-news-classifier-finetuned")
4. Launch the app
streamlit run app.py
