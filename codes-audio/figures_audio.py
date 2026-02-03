import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ====================== PATH ======================
CSV_PATH = r"C:\Users\ADmiN\Desktop\video_project\advanced_audio_features_corrected.csv"
SAVE_DIR = r"C:\Users\ADmiN\Desktop\video_project\reports\figures_audios"

# Création du dossier s'il n'existe pas
os.makedirs(SAVE_DIR, exist_ok=True)

# ====================== Load Data ======================
df = pd.read_csv(CSV_PATH)
print("✅ Données chargées depuis le CSV avec succès!")

# ====================== Figure 6 : Variabilité du pitch moyen entre vidéos ======================
# Extraction du nom de vidéo depuis la colonne chunk
df['video_name'] = df['chunk'].str.extract(r'(video\d+)')

if df['video_name'].nunique() > 1:
    plt.figure(figsize=(15, 8))
    
    pitch_per_video = df.groupby('video_name')['pitch_mean'].mean().reset_index()
    pitch_per_video = pitch_per_video.sort_values(by='pitch_mean')
    
    bars = plt.bar(range(len(pitch_per_video)), pitch_per_video['pitch_mean'], 
                  color='skyblue', edgecolor='navy', alpha=0.7, width=0.6)
    
    # Ajout des valeurs sur les barres
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('ID de la vidéo', fontsize=12)
    plt.ylabel('Pitch moyen (Hz)', fontsize=12)
    plt.title('Variabilité du pitch moyen entre vidéos', fontsize=14, pad=20)
    
    plt.xticks(range(len(pitch_per_video)), 
              pitch_per_video['video_name'], 
              rotation=45, ha='right')
    
    mean_pitch = pitch_per_video['pitch_mean'].mean()
    plt.axhline(y=mean_pitch, color='red', linestyle='--', linewidth=2,
               label=f'Moyenne générale: {mean_pitch:.1f} Hz')
    
    plt.legend(fontsize=10)
    plt.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    plt.tight_layout()
    
    # Sauvegarde du graphique
    plt.savefig(os.path.join(SAVE_DIR, "figure_pitch_per_video.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Statistiques
    print("\n" + "="*50)
    print("📊 Statistiques du Pitch:")
    print("="*50)
    print(f"Nombre de vidéos: {len(pitch_per_video)}")
    print(f"Moyenne générale: {mean_pitch:.2f} Hz")
    print(f"Écart-type: {pitch_per_video['pitch_mean'].std():.2f} Hz")
    print(f"Valeur minimale: {pitch_per_video['pitch_mean'].min():.1f} Hz")
    print(f"Valeur maximale: {pitch_per_video['pitch_mean'].max():.1f} Hz")
    print(f"Étendue: {pitch_per_video['pitch_mean'].max() - pitch_per_video['pitch_mean'].min():.1f} Hz")
    
    print("\n" + "="*50)
    print("📋 Détails par vidéo:")
    print("="*50)
    for idx, row in pitch_per_video.iterrows():
        video_chunks = df[df['video_name'] == row['video_name']].shape[0]
        print(f"{row['video_name']:10} | Pitch: {row['pitch_mean']:6.1f} Hz | Chunks: {video_chunks:2d}")
else:
    print("⚠️ Aucune variété de vidéos trouvée!")

# ====================== Figure 1 : Histogramme du pitch moyen ======================
plt.figure(figsize=(8,5))
plt.hist(df["pitch_mean"], bins=40, color='skyblue', edgecolor='black')
plt.title("Distribution du Pitch Moyen par Chunk")
plt.xlabel("Pitch moyen (Hz)")
plt.ylabel("Nombre de segments")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "figure_pitch_mean.png"))
plt.close()

# ====================== Figure 2 : Boxplot MFCC1 moyen (normalisé) ======================
plt.figure(figsize=(8,5))
sns.boxplot(y=df["mfcc_1_mean"], color='lightgreen')
plt.title("Distribution du MFCC1 moyen (normalisé)")
plt.ylabel("MFCC1 moyen")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "figure_mfcc1_mean.png"))
plt.close()

# ====================== Figure 3 : Spectral centroid ======================
plt.figure(figsize=(8,5))
plt.hist(df["spectral_centroid_mean"], bins=40, color='salmon', edgecolor='black')
plt.title("Distribution du Spectral Centroid")
plt.xlabel("Spectral Centroid (Hz)")
plt.ylabel("Nombre de segments")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "figure_spectral_centroid.png"))
plt.close()

# ====================== Figure 4 : Pauses par minute ======================
plt.figure(figsize=(8,5))
plt.hist(df["pauses_per_minute"], bins=30, color='violet', edgecolor='black')
plt.title("Distribution des Pauses par Minute")
plt.xlabel("Pauses/min")
plt.ylabel("Nombre de segments")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "figure_pauses_per_minute.png"))
plt.close()

# ====================== Figure 5 : Heatmap MFCCs moyens (MFCC1 à MFCC13) ======================
mfcc_cols = [f'mfcc_{i}_mean' for i in range(1,14)]
plt.figure(figsize=(12,6))
sns.heatmap(df[mfcc_cols].T, cmap="coolwarm", cbar=True)
plt.title("Heatmap des MFCCs Moyens par Chunk")
plt.xlabel("Index du segment")
plt.ylabel("Coefficient MFCC")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "figure_mfcc_heatmap.png"))
plt.close()

print("\n✅ Tous les graphiques ont été enregistrés dans :", SAVE_DIR)