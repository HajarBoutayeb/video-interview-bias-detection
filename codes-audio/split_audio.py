import os
from pydub import AudioSegment

input_dir = "full_audios"
output_dir = "audio_chunks"
chunk_duration = 15 * 1000  # 15 secondes

AudioSegment.converter = "C:\\Users\\ADmiN\\Desktop\\video_project\\ffmpeg\\ffmpeg.exe"
AudioSegment.ffprobe = "C:\\Users\\ADmiN\\Desktop\\video_project\\ffmpeg\\ffprobe.exe"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".mp3")):
        input_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]

        print(f"🔊 Lecture : {filename}")
        audio = AudioSegment.from_file(input_path)

        audio_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(audio_output_dir, exist_ok=True)

        for i in range(0, len(audio), chunk_duration):
            chunk = audio[i:i + chunk_duration]
            chunk_name = f"{base_name}_{i // chunk_duration:03d}.mp3"
            chunk_path = os.path.join(audio_output_dir, chunk_name)
            chunk.export(chunk_path, format="mp3")
        
        print(f"✅ Terminé : {filename}")
