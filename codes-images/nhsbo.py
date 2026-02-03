import pandas as pd
import numpy as np

# Charger le CSV
df = pd.read_csv("C:/Users/ADmiN/Desktop/video_project/advanced_audio_features_corrected.csv")

# Vérifier les colonnes MFCC
mfcc_cols = [col for col in df.columns if "mfcc" in col]
print(f"Colonnes MFCC trouvées : {mfcc_cols}")

# Recalculer les moyennes par segment
mfcc_means = df[mfcc_cols].mean()
mfcc_std = df[mfcc_cols].std()

print("\n=== Vérification MFCC ===")
for col in mfcc_cols:
    print(f"{col} -> moyenne : {mfcc_means[col]:.2f}, std : {mfcc_std[col]:.2f}")

# Optionnel : corriger MFCC1 si valeur trop extrême
if abs(mfcc_means["mfcc_1_mean"]) > 100:
    print("\n[!] MFCC1 semble anormal, recalcul avec normalisation min-max")
    df[mfcc_cols] = (df[mfcc_cols] - df[mfcc_cols].min()) / (df[mfcc_cols].max() - df[mfcc_cols].min())
    print("MFCC normalisés entre 0 et 1")
    mfcc_means = df[mfcc_cols].mean()
    print(f"Nouvelle moyenne MFCC1 : {mfcc_means['mfcc_1_mean']:.2f}")

# Sauvegarder fichier corrigé pour figures
df.to_csv("advanced_audio_features_corrected.csv", index=False)
print("\n✅ CSV corrigé sauvegardé pour figures.")
