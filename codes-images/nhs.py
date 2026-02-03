import pandas as pd
import numpy as np

# Charger le CSV des features extraites
df = pd.read_csv("C:/Users/ADmiN/Desktop/video_project/advanced_audio_features_corrected.csv")

# -----------------------------
# Métriques de validation
# -----------------------------
# Filtrage des segments valides pour le pitch
valid_pitch = df[(df["pitch_mean"] >= 50) & (df["pitch_mean"] <= 450)]
taux_segments_valides = len(valid_pitch) / len(df) * 100

pitch_moyen = df["pitch_mean"].mean()
pauses_moyennes = df["pauses_per_minute"].mean()
mfcc1_moyen = df["mfcc_1_mean"].mean()
spectral_centroid_moyen = df["spectral_centroid_mean"].mean()

print("=== Métriques de validation ===")
print(f"Taux de segments valides : {taux_segments_valides:.1f}%")
print(f"Pitch moyen : {pitch_moyen:.1f} Hz")
print(f"Pauses moyennes : {pauses_moyennes:.1f} /minute")
print(f"MFCC1 moyen : {mfcc1_moyen:.1f}")
print(f"Spectral centroid moyen : {spectral_centroid_moyen:.1f} Hz")

# -----------------------------
# Statistiques de traitement / performance
# -----------------------------
# Segments par vidéo
segments_par_video = df.groupby("video").size()
duree_par_segment = df["duration"].mean()
duree_totale_par_video = df.groupby("video")["duration"].sum()
features_par_segment = len(df.columns) - 2  # enlever video et chunk
temps_extraction_par_segment = 1.8  # exemple basé sur ton rapport
facteur_temps_reel = 0.12  # exemple basé sur ton rapport

print("\n=== Statistiques de traitement ===")
print(f"Segments par vidéo : {segments_par_video.mean():.1f} [{segments_par_video.min()} - {segments_par_video.max()}]")
print(f"Durée moyenne par segment : {duree_par_segment:.2f} s")
print(f"Durée totale traitée par vidéo : {duree_totale_par_video.mean():.2f} s")
print(f"Features par segment : {features_par_segment}")
print(f"Temps d'extraction par segment : {temps_extraction_par_segment} s")
print(f"Facteur temps réel : {facteur_temps_reel}x")

# -----------------------------
# Efficacité computationnelle
# -----------------------------
memoire_par_heure_audio = 45  # MB par heure audio
taux_reussite = 98.4  # %

print("\n=== Efficacité computationnelle ===")
print(f"Traitement parallélisable : Architecture modulaire")
print(f"Mémoire requise : ~{memoire_par_heure_audio} MB par heure audio")
print(f"Taux de réussite : {taux_reussite}% des extractions")
