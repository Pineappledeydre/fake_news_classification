import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import re
import nltk
import gensim
import string
from nltk.corpus import stopwords

# Load Model
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

# Load the trained model
model = BertClassifier()
model.load_state_dict(torch.load("models/bert_trained_model.pth", map_location=torch.device('cpu')))
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    return text

# Streamlit App
st.title("Fake News Detector with BERT")
user_input = st.text_area("Enter a tweet:")

if st.button("Analyze"):
    if user_input:
        cleaned_text = preprocess(user_input)
        encoding = tokenizer.encode_plus(
            cleaned_text,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        with torch.no_grad():
            prediction = model(input_ids, attention_mask).item()
        
        probability_real = 1 - prediction
        probability_fake = prediction
        
        st.write(f"### Probability of being Real: {probability_real:.2%}")
        st.write(f"### Probability of being Fake: {probability_fake:.2%}")
    else:
        st.write("Please enter a tweet to analyze.")
