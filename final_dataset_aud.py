import pandas as pd

# Chemins
labels_path = r"C:\Users\ADmiN\Desktop\video_project\dataset_final.xlsx"
features_path = r"C:\Users\ADmiN\Desktop\video_project\annotations\advanced_audio_features_fixed_colored.xlsx"
output_path = r"C:\Users\ADmiN\Desktop\video_project\labels_features_audios.xlsx"

# Chargement des fichiers Excel
labels_df = pd.read_excel(labels_path)
features_df = pd.read_excel(features_path)

print("✅ Colonnes des labels:", labels_df.columns.tolist())
print("✅ Colonnes des features:", features_df.columns.tolist())

# ---- Préparation des colonnes pour compatibilité ----
# Renommer 'file' → 'clip_name' dans features
if "file" in features_df.columns:
    features_df = features_df.rename(columns={"file": "clip_name"})

# Fusion sur clip_name
merged_df = pd.merge(labels_df, features_df, on="clip_name", how="left")

# Organisation des colonnes: textes + labels + features
base_cols = ['filename', 'line_number', 'text', 'label', 'bias_type', 'severity',
             'video_id', 'clip_name', 'clip_text']
feature_cols = [col for col in merged_df.columns if col not in base_cols]
merged_df = merged_df[base_cols + feature_cols]

# Sauvegarde du dataset final
merged_df.to_excel(output_path, index=False)

print("✅ Terminé! Dataset combiné sauvegardé dans:", output_path)