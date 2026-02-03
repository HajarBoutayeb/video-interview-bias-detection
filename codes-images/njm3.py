import pandas as pd

# ======== Charger les deux fichiers ========
df1 = pd.read_excel(r"C:\Users\ADmiN\Desktop\video_project\dataset_final.xlsx")
df2 = pd.read_excel(r"C:\Users\ADmiN\Desktop\video_project\advanced_audio_features_fixed_colored.xlsx")

# ======== Fusionner sur clip_name = file ========
merged_df = pd.merge(df1, df2, left_on="clip_name", right_on="file", how="inner")

# ======== Sauvegarder le fichier fusionné ========
output_path = r"C:\Users\ADmiN\Desktop\video_project\dataset_merged.xlsx"
merged_df.to_excel(output_path, index=False)

print("✅ Fusion terminée !")
print("Dimensions du dataset fusionné :", merged_df.shape)
