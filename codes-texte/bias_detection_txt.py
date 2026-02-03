import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
import numpy as np

# ======== Chemins ========
transcripts_path = r"C:\Users\ADmiN\Desktop\video_project\Interview_Transcripts_For_Bias"
labels_path = r"C:\Users\ADmiN\Desktop\video_project\annotations\text labels.xlsx"

# ======== Charger les étiquettes ========
labels_df = pd.read_excel(labels_path)

# ======== Charger les transcriptions ========
all_texts = []
all_labels = []

for file in sorted(os.listdir(transcripts_path)):
    if file.endswith(".txt"):
        with open(os.path.join(transcripts_path, file), "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            all_texts.extend(lines)
        file_label_row = labels_df[labels_df['filename'] == file]
        if not file_label_row.empty:
            label = file_label_row['bias_type'].values[0]
        else:
            label = 'Unknown'
        all_labels.extend([label]*len(lines))

# ======== Encoder les étiquettes ========
le = LabelEncoder()
y = le.fit_transform(all_labels)

# ======== Équilibrer le dataset ========
df_ml = pd.DataFrame({'text': all_texts, 'label': y})
max_size = df_ml['label'].value_counts().max()
lst = [df_ml]
for class_index, group in df_ml.groupby('label'):
    lst.append(group.sample(max_size - len(group), replace=True))
df_ml_balanced = pd.concat(lst)

all_texts = df_ml_balanced['text'].tolist()
y = df_ml_balanced['label'].values

# ======== Embeddings BERT ========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model.to(device)
model.eval()

def get_embeddings(texts, batch_size=32):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            out = model(**enc)
            # Prendre le pooling moyen
            emb = out.last_hidden_state.mean(dim=1)
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)

print("Génération des embeddings BERT...")
X = get_embeddings(all_texts)

# ======== Division entraînement/test ========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ======== Modèles ========
models = {
    "RandomForest": RandomForestClassifier(n_estimators=700, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=700, learning_rate=0.05, max_depth=5, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=2000, solver='saga', multi_class='multinomial', random_state=42)
}

# ======== Entraîner et Évaluer ========
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    results.append([name, round(acc,3), round(prec,3), round(rec,3), round(f1,3)])

    # Afficher en style empilé
    print("\n============================")
    print(f"Modèle         : {name}")
    print(f"Précision      : {round(acc,3)}")
    print(f"Précision      : {round(prec,3)}")
    print(f"Rappel         : {round(rec,3)}")
    print(f"F1-score       : {round(f1,3)}")
    print("============================")

# ======== Sauvegarder les résultats ========
results_df = pd.DataFrame(results, columns=["Modèle","Précision","Précision","Rappel","F1-score"])
results_df.to_excel(os.path.join(transcripts_path, "models_evaluation_BERT.xlsx"), index=False)
print("\nRésultats sauvegardés dans 'models_evaluation_BERT.xlsx'")