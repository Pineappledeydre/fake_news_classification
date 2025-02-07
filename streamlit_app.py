import streamlit as st
import torch
import numpy as np
import re
import string
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset
import torch.nn as nn

# ========== MODEL CLASS ==========
class BertClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)  # Binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        fc_output = self.fc(dropout_output)
        return self.sigmoid(fc_output)

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    model = BertClassifier()
    github_url = "https://raw.githubusercontent.com/Pineappledeydre/fake_news_classification/main/models/trained_model.pth"
    model_path = "trained_model.pth"

    # Download model from GitHub
    import urllib.request
    urllib.request.urlretrieve(github_url, model_path)
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ========== TEXT PREPROCESSING ==========
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    return text

def encode_text(text, tokenizer, max_length=128):
    encoding = tokenizer.encode_plus(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return encoding["input_ids"], encoding["attention_mask"]

# ========== STREAMLIT APP ==========
st.title("Fake News Detection with BERT")
st.write("Enter a news text, and the model will predict whether it is **real or fake**.")

user_input = st.text_area("Enter News Text")

if st.button("Predict"):
    if user_input:
        processed_input = clean_text(user_input)
        st.write(f"**Processed Text:** {processed_input}")

        input_ids, attention_mask = encode_text(processed_input, tokenizer)

        with torch.no_grad():
            output = model(input_ids, attention_mask)
            prediction = (output > 0.5).float().item()

        if prediction == 1:
            st.success("This is **Real News**!")
        else:
            st.error("This is **Fake News**!")
    else:
        st.warning("Please enter some text to analyze.")
