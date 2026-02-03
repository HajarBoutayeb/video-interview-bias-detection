import pandas as pd

# Charger les données
df = pd.read_excel(r"C:\Users\ADmiN\Desktop\video_project\annotations\faces_annotations.xlsx")

# Nettoyer face_detected : mettre 1 si valeur indique un visage, 0 sinon
df['face_detected'] = df['face_detected'].apply(lambda x: 1 if str(x).lower() in ['1', 'yes', '✅ yes'] else 0)

# ===================== Statistiques globales =====================
total_images = df.shape[0]
total_faces = df['face_detected'].sum()
faces_percent = total_faces / total_images * 100

# Nettoyer et convertir âge en float
df['age'] = pd.to_numeric(df['age'], errors='coerce')
age_mean = df['age'].mean()
age_std = df['age'].std()

# Genre
df['gender'] = df['gender'].astype(str)
gender_counts = df['gender'].value_counts()
gender_percent = df['gender'].value_counts(normalize=True) * 100

# Emotion
df['emotion'] = df['emotion'].astype(str)
emotion_counts = df['emotion'].value_counts()
emotion_percent = df['emotion'].value_counts(normalize=True) * 100

# Race
df['race'] = df['race'].astype(str)
race_counts = df['race'].value_counts()
race_percent = df['race'].value_counts(normalize=True) * 100

# ===================== Affichage =====================
print(f"Nombre total d’images analysées : {total_images}")
print(f"Nombre de visages détectés : {total_faces} ({faces_percent:.1f}%)")
print(f"Âge moyen estimé : {age_mean:.1f} ans (σ = {age_std:.1f})\n")

print("Répartition par genre :")
for g in gender_counts.index:
    print(f"{g} : {gender_counts[g]} ({gender_percent[g]:.1f}%)")

print("\nDistribution des émotions :")
for e in emotion_counts.index:
    print(f"{e} : {emotion_counts[e]} ({emotion_percent[e]:.1f}%)")

print("\nRépartition par race prédite :")
for r in race_counts.index:
    print(f"{r} : {race_counts[r]} ({race_percent[r]:.1f}%)")
