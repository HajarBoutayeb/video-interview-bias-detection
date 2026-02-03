import os

# Dossier contenant les clips
clips_dir = r"C:\Users\ADmiN\Desktop\video_project\clips"
# Dossier où seront placées les images
output_dir = r"C:\Users\ADmiN\Desktop\video_project\grouped_images"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(clips_dir):
    if filename.endswith(".mp4"):
        clip_path = os.path.join(clips_dir, filename)
        base_name = os.path.splitext(filename)[0]

        # Création d'un dossier spécifique pour les images de chaque clip
        clip_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(clip_output_dir, exist_ok=True)

        # Extraction d'une image par seconde en utilisant ffmpeg
        os.system(f'ffmpeg -i "{clip_path}" -vf fps=1 "{clip_output_dir}/frame_%03d.jpg"')

print("✅ Extraction des images (une image par seconde) terminée pour tous les clips et sauvegardée dans grouped_images")