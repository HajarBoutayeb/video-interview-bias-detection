import os
import pandas as pd
import re
import string
from difflib import get_close_matches

# =========================
# Chemins
# =========================
labels_path = r"C:\Users\ADmiN\Desktop\video_project\annotations\text labels.xlsx"
transcripts_folder = r"C:\Users\ADmiN\Desktop\video_project\Interview_Transcripts_For_Bias"
output_path = r"C:\Users\ADmiN\Desktop\video_project\final_dataset_final_precise_B.xlsx"

# =========================
# Charger les étiquettes
# =========================
labels_df = pd.read_excel(labels_path)

# =========================
# Fonction auxiliaire : nettoyer le texte
# =========================
def clean_text(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text

# =========================
# Lire les transcriptions et diviser en clips (cumulatif)
# =========================
clips_dict = {}  # {video_id: {clip_name: [lines]}}
for file in sorted(os.listdir(transcripts_folder)):
    if file.endswith(".txt"):
        video_id = os.path.splitext(file)[0]
        with open(os.path.join(transcripts_folder, file), "r", encoding="utf-8") as f:
            content = f.read()
            # Diviser par les marqueurs de clip --- videoX_YYY ---
            parts = re.split(r'---\s*(video\d+_\d+\.mp3)\s*---', content)
            clip_lines = {}
            for i in range(1, len(parts), 2):
                clip_name = parts[i].strip()
                text = parts[i+1].strip().split("\n")
                text = [clean_text(line) for line in text if line.strip()]
                clip_lines[clip_name] = text
            # Fusionner cumulativement
            if video_id in clips_dict:
                clips_dict[video_id].update(clip_lines)
            else:
                clips_dict[video_id] = clip_lines

# =========================
# Assigner chaque ligne à son clip automatiquement en utilisant la correspondance floue
# =========================
labels_df["video_id"] = labels_df["filename"].apply(lambda x: os.path.splitext(x)[0])

def assign_clip_transcript_fuzzy(row):
    vid = row['video_id']
    line_text = clean_text(row['text'])
    if vid in clips_dict:
        for clip_name, lines in clips_dict[vid].items():
            match = get_close_matches(line_text, lines, n=1, cutoff=0.7)
            if match:
                return pd.Series([clip_name, match[0]])
    return pd.Series([None, None])

labels_df[['clip_name', 'clip_text']] = labels_df.apply(assign_clip_transcript_fuzzy, axis=1)

# =========================
# Réorganiser les colonnes
# =========================
cols_order = ['filename', 'line_number', 'text', 'label', 'bias_type', 'severity',
              'video_id', 'clip_name', 'clip_text']
labels_df = labels_df[cols_order]

# =========================
# Sauvegarder le fichier Excel final
# =========================
labels_df.to_excel(output_path, index=False)
print("✅ Terminé ! Chaque ligne a maintenant son clip précis et le texte correspondant :", output_path)