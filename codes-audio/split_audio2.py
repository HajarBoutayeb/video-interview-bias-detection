import os
from pydub import AudioSegment

AudioSegment.converter = r"C:\Users\ADmiN\Desktop\video_project\ffmpeg\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\Users\ADmiN\Desktop\video_project\ffmpeg\ffprobe.exe"

full_audio_dir = "full_audios"
split_audio_dir = "audio_chunks"

os.makedirs(split_audio_dir, exist_ok=True)

segment_duration_ms = 15 * 1000  # 15 ثانية

for filename in os.listdir(full_audio_dir):
    if filename.endswith(".webm"):
        audio_path = os.path.join(full_audio_dir, filename)
        audio = AudioSegment.from_file(audio_path, format="webm")

        duration_ms = len(audio)
        num_segments = duration_ms // segment_duration_ms

        base_name = os.path.splitext(filename)[0]

        for i in range(num_segments):
            start_ms = i * segment_duration_ms
            end_ms = start_ms + segment_duration_ms

            segment = audio[start_ms:end_ms]
            out_path = os.path.join(split_audio_dir, f"{base_name}_clip{i+1}.wav")
            segment.export(out_path, format="wav")
            print(f"صدّرنا: {out_path}")

print("✅ تقسيم الصوتيات تم بنجاح!")
