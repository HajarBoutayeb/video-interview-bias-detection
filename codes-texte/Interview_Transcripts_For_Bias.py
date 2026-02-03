import os
import re

# Dossier d'entrée (transcriptions brutes)
input_dir = r"C:\Users\ADmiN\Desktop\video_project\transcripts"
# Dossier de sortie (transcriptions nettoyées et structurées)
output_dir = r"C:\Users\ADmiN\Desktop\video_project\Interview_Transcripts_For_Bias"
os.makedirs(output_dir, exist_ok=True)

# Segmentation simple par ponctuation
def segment_text(text):
    # Division sur . ? ! en les conservant avec la phrase
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
            content = f.read()

        # Division des segments selon les balises "--- videoX_000 ---"
        blocks = re.split(r'---\s*video\d+_\d+\s*---', content, flags=re.IGNORECASE)
        matches = re.findall(r'---\s*video\d+_\d+\s*---', content, flags=re.IGNORECASE)

        cleaned_blocks = []
        for i, block in enumerate(blocks[1:]):  # block[0] est généralement vide avant le premier match
            video_id = matches[i].strip()
            sentences = segment_text(block)
            cleaned_text = video_id + "\n" + "\n".join(sentences) + "\n"
            cleaned_blocks.append(cleaned_text)

        # Sauvegarde du fichier structuré
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned_blocks))

print("✅ Transcriptions nettoyées et sauvegardées dans :", output_dir)