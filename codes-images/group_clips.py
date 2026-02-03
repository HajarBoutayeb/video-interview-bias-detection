import os
import shutil

# Dossier contenant tous les clips
clips_dir = "clips"

# Parcourir tous les fichiers dans le dossier
for filename in os.listdir(clips_dir):
    if filename.endswith(".mp4"):
        # Extraction du nom de la vidéo originale (par exemple: video1 depuis video1_000.mp4)
        base_name = filename.split("_")[0]
        base_folder = os.path.join(clips_dir, base_name)

        # Création d'un dossier spécifique pour la vidéo s'il n'existe pas
        os.makedirs(base_folder, exist_ok=True)

        # Définition des chemins ancien et nouveau
        old_path = os.path.join(clips_dir, filename)
        new_path = os.path.join(base_folder, filename)

        # Déplacement du fichier dans le dossier approprié
        shutil.move(old_path, new_path)

print("✅ Les clips ont été regroupés dans leurs dossiers selon leur nom.")