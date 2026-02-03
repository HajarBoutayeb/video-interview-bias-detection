import os
import whisper

model = whisper.load_model("large-v3")

audio_chunks_dir = "audio_chunks"
output_texts_dir = "transcripts"
os.makedirs(output_texts_dir, exist_ok=True)

target_language = "en"

for video_folder in os.listdir(audio_chunks_dir):
    video_folder_path = os.path.join(audio_chunks_dir, video_folder)
    if os.path.isdir(video_folder_path):
        print(f"Processing folder: {video_folder}")
        
        transcript_path = os.path.join(output_texts_dir, f"{video_folder}.txt")

        with open(transcript_path, "w", encoding="utf-8") as transcript_file:
            audio_files = sorted([f for f in os.listdir(video_folder_path) if f.endswith(".mp3")])
            
            for audio_file in audio_files:
                audio_path = os.path.join(video_folder_path, audio_file)
                print(f"  Transcribing {audio_file} ...")
                
                result = model.transcribe(
                    audio_path,
                    language=target_language,  # Force English output
                    fp16=False  # For CPU usage
                )
                
                text = result["text"].strip()
                transcript_file.write(f"--- {audio_file} ---\n")
                transcript_file.write(text + "\n\n")
        
        print(f"Transcript saved to {transcript_path}\n")

print("All done!")
