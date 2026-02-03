import os
import subprocess
import shutil
import re

clips_root = "clips"  
output_root = "grouped_images"  
os.makedirs(output_root, exist_ok=True)

for folder in os.listdir(clips_root):
    folder_path = os.path.join(clips_root, folder)

    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".mp4"):
                clip_path = os.path.join(folder_path, filename)

                match = re.match(r"(.+?)\.f\d+_clip(\d+)\.mp4", filename)
                if not match:
                    print(f"Nom de fichier non reconnu : {filename}")
                    continue

                base_video = match.group(1)
                clip_num = int(match.group(2))

                new_clip_name = f"{base_video}_clip_{clip_num:03d}"
                video_folder = os.path.join(output_root, base_video)
                clip_folder = os.path.join(video_folder, new_clip_name)
                os.makedirs(clip_folder, exist_ok=True)

                command = [
                    "ffmpeg",
                    "-i", clip_path,
                    "-vf", "fps=1",
                    os.path.join(clip_folder, "frame_%03d.jpg")
                ]
                print(f"Extraction des images depuis : {filename}")
                subprocess.run(command)

print("✅ Extraction et organisation terminées dans 'grouped_images'.")
