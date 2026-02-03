import os
import subprocess

# Chemin du dossier contenant les fichiers
base_dir = r"C:\Users\ADmiN\Desktop\video_project\audio_chunks"

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith((".f251", ".f234")):
            input_path = os.path.join(root, file)
            output_path = os.path.splitext(input_path)[0] + ".mp3"

            # Ne pas convertir si le mp3 existe déjà
            if not os.path.exists(output_path):
                subprocess.run([
                    "ffmpeg", "-y", "-i", input_path, output_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

                print(f"✅ {file} → {os.path.basename(output_path)}")

print("🎯 La conversion est terminée pour tous les fichiers.")