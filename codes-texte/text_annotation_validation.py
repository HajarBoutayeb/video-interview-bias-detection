import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch

# Chargement du dataset
df = pd.read_excel("C:/Users/ADmiN/Desktop/video_project/annotations/text labels.xlsx")

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(df["context"].astype(str))  # ici on prend la colonne texte
print("TF-IDF vectorization : shape =", X_tfidf.shape)

# BERT embeddings
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertModel.from_pretrained("bert-base-multilingual-cased")

def get_bert_embeddings(texts):
    embeddings = []
    for t in texts:
        inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:,0,:]
        embeddings.append(cls_embedding.squeeze().numpy())
    return embeddings

bert_embeddings = get_bert_embeddings(df["context"].astype(str).tolist())
print("BERT embeddings : nombre de vecteurs =", len(bert_embeddings))

# Statistiques du dataset
total_phrases = len(df)
bias_distribution = df["bias_type"].value_counts()
severity_distribution = df["severity"].value_counts()

print("\nStatistiques du dataset :")
print(f"Phrases totales : {total_phrases}")
print("\nDistribution par type de biais (top 10) :")
print(bias_distribution.head(10))
print("\nDistribution par niveau de sévérité :")
print(severity_distribution)
