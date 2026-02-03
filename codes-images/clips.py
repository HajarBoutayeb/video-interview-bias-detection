import os
import cv2

# Chemin contenant les vidéos originales
videos_dir = r"C:\Users\ADmiN\Desktop\video_project\videos"
# Chemin où seront placés les clips
clips_dir = r"C:\Users\ADmiN\Desktop\video_project\clips"
os.makedirs(clips_dir, exist_ok=True)

# Parcourir toutes les vidéos
for filename in os.listdir(videos_dir):
    if filename.endswith(".mp4"):
        video_path = os.path.join(videos_dir, filename)
        base_name = os.path.splitext(filename)[0]

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames // fps

        clip_length = 15  # Chaque clip dure 15 secondes
        clip_index = 0

        for start_time in range(0, duration, clip_length):
            end_time = min(start_time + clip_length, duration)

            # Nom du clip
            clip_name = f"{base_name}_{clip_index:03d}.mp4"
            clip_path = os.path.join(clips_dir, clip_name)

            # ffmpeg pour découper la vidéo
            os.system(f'ffmpeg -i "{video_path}" -ss {start_time} -to {end_time} -c copy "{clip_path}"')

            clip_index += 1

        cap.release()

print("✅ Les vidéos ont été divisées en clips de 15 secondes")