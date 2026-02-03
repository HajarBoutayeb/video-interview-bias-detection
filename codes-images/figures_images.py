import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= PATH =================
EXCEL_PATH = r"C:\Users\ADmiN\Desktop\video_project\annotations\faces_annotations.xlsx"
SAVE_DIR = r"C:\Users\ADmiN\Desktop\video_project\reports\figures_images"

# ================= Create folder if not exists =================
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= Load Data =================
df = pd.read_excel(EXCEL_PATH)

print(df.head())
print(df.columns)

# ================= Figures =================

# 1) Distribution par Age
plt.figure(figsize=(8,5))
sns.histplot(df['age'], bins=15, kde=True)
plt.title("Distribution des âges détectés")
plt.xlabel("Âge")
plt.ylabel("Fréquence")
plt.savefig(os.path.join(SAVE_DIR, "distribution_age.png"))
plt.close()

# 2) Distribution par Genre
plt.figure(figsize=(6,4))
df['gender'].value_counts().plot(kind='bar', color=['skyblue','salmon'])
plt.title("Distribution par genre")
plt.xlabel("Genre")
plt.ylabel("Nombre d'occurrences")
plt.savefig(os.path.join(SAVE_DIR, "distribution_genre.png"))
plt.close()

# 3) Distribution par Émotion
plt.figure(figsize=(8,5))
df['emotion'].value_counts().plot(kind='bar', color='lightgreen')
plt.title("Distribution des émotions détectées")
plt.xlabel("Émotion")
plt.ylabel("Nombre d'occurrences")
plt.savefig(os.path.join(SAVE_DIR, "distribution_emotion.png"))
plt.close()

# 4) Distribution par Race
plt.figure(figsize=(8,5))
df['race'].value_counts().plot(kind='bar', color='orange')
plt.title("Distribution des races détectées")
plt.xlabel("Race")
plt.ylabel("Nombre d'occurrences")
plt.savefig(os.path.join(SAVE_DIR, "distribution_race.png"))
plt.close()

# 5) Relation Age - Emotion (Boxplot)
plt.figure(figsize=(10,6))
sns.boxplot(x='emotion', y='age', data=df)
plt.title("Variation de l'âge selon l'émotion")
plt.xticks(rotation=45)
plt.savefig(os.path.join(SAVE_DIR, "age_vs_emotion.png"))
plt.close()

# 6) Relation Genre - Emotion (Heatmap)
pivot = pd.crosstab(df['gender'], df['emotion'])
plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Matrice Genre vs Émotion")
plt.savefig(os.path.join(SAVE_DIR, "genre_vs_emotion_heatmap.png"))
plt.close()

print("✅ Tous les graphiques ont été enregistrés dans :", SAVE_DIR)
