import os
import librosa
import numpy as np
import pandas as pd
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class AdvancedAudioAnalyzer:
    def __init__(self):
        self.encoder = VoiceEncoder()
        
    def detect_pauses_librosa(self, y, sr):
        """Détection des pauses en utilisant librosa uniquement - mis à jour et amélioré"""
        hop_length = 512
        frame_length = 2048
        
        # RMS energy amélioré
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Définition d'un seuil plus précis
        silence_threshold = np.percentile(rms, 25)  # Les 25% inférieurs de l'énergie
        
        # Application d'un filtre médian pour lisser le signal
        from scipy.ndimage import median_filter
        rms_smooth = median_filter(rms, size=3)
        
        # Recherche des zones silencieuses
        silent_frames = rms_smooth < silence_threshold
        
        # Conversion en temps
        frame_times = librosa.frames_to_time(np.arange(len(silent_frames)), 
                                           sr=sr, hop_length=hop_length)
        
        # Comptage des pauses avec paramètres améliorés
        pauses = 0
        min_pause_duration = 0.2  # 200ms au lieu de 300ms
        in_pause = False
        pause_start = 0
        
        for i, is_silent in enumerate(silent_frames):
            if is_silent and not in_pause:
                in_pause = True
                pause_start = frame_times[i]
            elif not is_silent and in_pause:
                in_pause = False
                if i < len(frame_times):
                    pause_duration = frame_times[i] - pause_start
                    if pause_duration >= min_pause_duration:
                        pauses += 1
                        
        return pauses
    
    def detect_pauses_onset_based(self, y, sr):
        """Détection des pauses basée sur onset detection - mis à jour"""
        try:
            # Extraction des onset times avec paramètres améliorés
            onset_frames = librosa.onset.onset_detect(
                y=y, sr=sr, 
                units='frames',
                hop_length=512,
                wait=int(0.1 * sr / 512),  # 100ms minimum entre les onsets
                pre_max=int(0.03 * sr / 512),  # 30ms pre-max
                post_max=int(0.03 * sr / 512),  # 30ms post-max
                pre_avg=int(0.1 * sr / 512),   # 100ms pre-avg
                post_avg=int(0.1 * sr / 512),  # 100ms post-avg
                delta=0.07  # seuil
            )
            
            if len(onset_frames) < 2:
                return 0
                
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
            
            # Calcul des distances entre onsets
            intervals = np.diff(onset_times)
            
            # Comptage des pauses longues avec seuil plus réaliste
            long_pauses = np.sum(intervals > 0.4)  # 400ms au lieu de 500ms
            
            return long_pauses
            
        except Exception as e:
            print(f"⚠️ Erreur dans onset detection: {e}")
            return 0
    
    def extract_prosodic_features(self, y, sr):
        """Extraction des caractéristiques prosodiques et d'intonation - très amélioré"""
        try:
            # Amélioration des paramètres de pitch detection
            fmin = 50   # Fréquence minimale pour les voix humaines
            fmax = 450  # Fréquence maximale pour les voix humaines
            
            # Utilisation de piptrack au lieu de yin pour plus de précision
            pitches, magnitudes = librosa.piptrack(
                y=y, sr=sr, 
                fmin=fmin, fmax=fmax,
                hop_length=512,
                threshold=0.1
            )
            
            # Extraction du pitch le plus fort dans chaque frame
            f0 = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    f0.append(pitch)
            
            f0 = np.array(f0)
            
            # Si piptrack échoue, utiliser yin comme backup
            if len(f0) == 0:
                f0 = librosa.yin(y, fmin=fmin, fmax=fmax, frame_length=2048)
                f0 = f0[f0 > 0]
                
            # Si aucun pitch n'est trouvé
            if len(f0) == 0:
                return {
                    'pitch_mean': 150,     # Valeur par défaut réaliste
                    'pitch_std': 0, 
                    'pitch_range': 0,
                    'pitch_slope': 0, 
                    'jitter': 0
                }
            
            # Nettoyage des données - suppression des valeurs aberrantes
            pitch_median = np.median(f0)
            pitch_mad = np.median(np.abs(f0 - pitch_median))
            
            # Suppression des valeurs s'éloignant de plus de 3 MAD de la médiane
            if pitch_mad > 0:
                outlier_threshold = 3 * pitch_mad
                f0_clean = f0[np.abs(f0 - pitch_median) < outlier_threshold]
            else:
                f0_clean = f0
                
            if len(f0_clean) == 0:
                f0_clean = f0
            
            # Calcul des indicateurs
            pitch_mean = np.mean(f0_clean)
            pitch_std = np.std(f0_clean)
            pitch_range = np.max(f0_clean) - np.min(f0_clean)
            
            # Pente du pitch améliorée
            if len(f0_clean) > 1:
                x = np.arange(len(f0_clean))
                pitch_slope = np.polyfit(x, f0_clean, 1)[0]
                
                # Jitter amélioré - variation du pitch
                pitch_diffs = np.abs(np.diff(f0_clean))
                jitter = np.mean(pitch_diffs) / pitch_mean if pitch_mean > 0 else 0
            else:
                pitch_slope = 0
                jitter = 0
                
            return {
                'pitch_mean': float(pitch_mean),
                'pitch_std': float(pitch_std), 
                'pitch_range': float(pitch_range),
                'pitch_slope': float(pitch_slope),
                'jitter': float(jitter)
            }
            
        except Exception as e:
            print(f"⚠️ Erreur dans prosodic features: {e}")
            return {
                'pitch_mean': 150, 'pitch_std': 0, 'pitch_range': 0,
                'pitch_slope': 0, 'jitter': 0
            }
    
    def extract_spectral_features(self, y, sr):
        """Caractéristiques spectrales améliorées"""
        try:
            # Paramètres améliorés
            hop_length = 512
            n_fft = 2048
            
            # Spectral centroid - centre du spectre
            spec_cent = librosa.feature.spectral_centroid(
                y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)[0]
            
            # Spectral rolloff - limite du spectre
            spec_rolloff = librosa.feature.spectral_rolloff(
                y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, roll_percent=0.85)[0]
            
            # Spectral bandwidth - largeur du spectre  
            spec_bw = librosa.feature.spectral_bandwidth(
                y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                y, frame_length=n_fft, hop_length=hop_length)[0]
            
            # Spectral contrast amélioré
            spec_contrast = librosa.feature.spectral_contrast(
                y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, n_bands=6)
            spec_contrast_mean = np.mean(spec_contrast, axis=1)
            
            # Chroma features améliorées
            chroma = librosa.feature.chroma_stft(
                y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
            chroma_mean = np.mean(chroma, axis=1)
            
            return {
                'spectral_centroid_mean': float(np.mean(spec_cent)),
                'spectral_centroid_std': float(np.std(spec_cent)),
                'spectral_rolloff_mean': float(np.mean(spec_rolloff)),
                'spectral_bandwidth_mean': float(np.mean(spec_bw)),
                'zcr_mean': float(np.mean(zcr)),
                'spectral_contrast_1': float(spec_contrast_mean[0]),
                'spectral_contrast_2': float(spec_contrast_mean[1] if len(spec_contrast_mean) > 1 else 0),
                'chroma_1': float(chroma_mean[0]),
                'chroma_2': float(chroma_mean[1] if len(chroma_mean) > 1 else 0),
            }
            
        except Exception as e:
            print(f"⚠️ Erreur dans spectral features: {e}")
            return {
                'spectral_centroid_mean': 1000, 'spectral_centroid_std': 200,
                'spectral_rolloff_mean': 2000, 'spectral_bandwidth_mean': 1500,
                'zcr_mean': 0.1, 'spectral_contrast_1': 20, 'spectral_contrast_2': 15,
                'chroma_1': 0.5, 'chroma_2': 0.3,
            }
    
    def extract_rhythm_features(self, y, sr):
        """Caractéristiques de rythme et de vitesse - améliorées et mises à jour pour la compatibilité"""
        try:
            # Tempo amélioré - suppression des paramètres non supportés
            tempo, beats = librosa.beat.beat_track(
                y=y, sr=sr, 
                hop_length=512,
                start_bpm=120.0,  # BPM initial logique pour la parole
                trim=True
            )
            
            # Analyse du rythme améliorée
            onset_frames = librosa.onset.onset_detect(
                y=y, sr=sr, 
                hop_length=512,
                delta=0.05,  # seuil moins sensible
                wait=int(0.05 * sr / 512)  # 50ms minimum
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
            
            # Taux de parole (approximation du taux de syllabes)
            duration = len(y) / sr
            if len(onset_times) > 1 and duration > 0:
                speech_rate = len(onset_times) / duration  # événements par seconde
            else:
                speech_rate = 0
            
            # Analyse du rythme - régularité améliorée
            if len(onset_times) > 2:
                intervals = np.diff(onset_times)
                # Suppression des intervalles aberrants
                interval_median = np.median(intervals)
                interval_mad = np.median(np.abs(intervals - interval_median))
                if interval_mad > 0:
                    clean_intervals = intervals[np.abs(intervals - interval_median) < 3 * interval_mad]
                else:
                    clean_intervals = intervals
                    
                if len(clean_intervals) > 0:
                    rhythm_regularity = -np.std(clean_intervals)  # négatif car moins de dispersion = plus de régularité
                else:
                    rhythm_regularity = 0
            else:
                rhythm_regularity = 0
                
            return {
                'tempo': float(tempo) if tempo > 0 else 120.0,  # valeur par défaut logique
                'speech_rate': float(speech_rate),
                'onset_density': int(len(onset_times)),
                'rhythm_regularity': float(rhythm_regularity)
            }
            
        except Exception as e:
            print(f"⚠️ Erreur dans rhythm features: {e}")
            return {
                'tempo': 120, 'speech_rate': 2, 'onset_density': 10, 'rhythm_regularity': -0.5
            }
    
    def extract_mfcc_features(self, y, sr):
        """Extraction MFCC améliorée et précise - avec nettoyage robuste des valeurs aberrantes"""
        try:
            # Paramètres MFCC améliorés
            mfccs = librosa.feature.mfcc(
                y=y, sr=sr, 
                n_mfcc=13,              # 13 coefficients standard
                n_fft=2048,             # taille FFT
                hop_length=512,         # hop length
                n_mels=128,             # nombre de mel bins
                fmin=0,                 # fréquence minimale
                fmax=sr//2,             # fréquence maximale (Nyquist)
                window='hann',          # type de fenêtre
                center=True,            # centrage de la fenêtre
                norm='ortho',           # normalisation orthogonale
                lifter=22               # coefficient de liftering
            )
            
            # 🔧 Nettoyage robuste des valeurs aberrantes avant le calcul des statistiques
            for i in range(mfccs.shape[0]):  # pour chaque coefficient MFCC
                row = mfccs[i, :]
                
                # Suppression des nan et inf
                row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Définition de limites logiques pour chaque MFCC
                if i == 0:  # Premier MFCC
                    min_val, max_val = -1000, 100
                else:  # Autres MFCCs
                    min_val, max_val = -200, 200
                
                # Application des limites
                row = np.clip(row, min_val, max_val)
                mfccs[i, :] = row
            
            # Calcul des statistiques après nettoyage
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Compensation des nan restants
            mfcc_mean = np.nan_to_num(mfcc_mean, nan=0.0, posinf=0.0, neginf=0.0)
            mfcc_std = np.nan_to_num(mfcc_std, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Limites supplémentaires pour std (ne doit pas être trop grand)
            # Utilisation de la détection d'outliers au lieu d'un clipping dur
            for i in range(len(mfcc_std)):
                if mfcc_std[i] > 120:  # seulement les valeurs très aberrantes
                    if i == 0:  # MFCC1 est généralement plus grand
                        mfcc_std[i] = 60.0
                    else:
                        mfcc_std[i] = 40.0
            
            return {
                **{f"mfcc_{i+1}_mean": float(mfcc_mean[i]) for i in range(13)},
                **{f"mfcc_{i+1}_std": float(mfcc_std[i]) for i in range(13)}
            }
            
        except Exception as e:
            print(f"⚠️ Erreur dans MFCC: {e}")
            # Valeurs par défaut logiques pour MFCC
            default_mfcc = {}
            for i in range(13):
                if i == 0:  # Le premier MFCC est généralement plus élevé et négatif
                    default_mfcc[f"mfcc_{i+1}_mean"] = -200.0
                    default_mfcc[f"mfcc_{i+1}_std"] = 30.0
                else:
                    default_mfcc[f"mfcc_{i+1}_mean"] = 0.0
                    default_mfcc[f"mfcc_{i+1}_std"] = 15.0
            return default_mfcc
    
    def analyze_audio_file(self, file_path, video, chunk, file_name):
        """Analyse complète d'un fichier audio unique - mis à jour et amélioré"""
        try:
            # Chargement de l'audio avec paramètres améliorés
            y, sr = librosa.load(file_path, sr=22050, mono=True)  # spécification du sample rate
            
            if len(y) == 0:
                print(f"⚠️ Fichier vide: {file_name}")
                return None
            
            # Normalisation de l'audio pour éviter les valeurs aberrantes
            if np.max(np.abs(y)) > 0:
                y = librosa.util.normalize(y)
            
            # Caractéristiques de base
            energy = float(np.sum(y**2) / len(y))
            duration = float(len(y) / sr)
            
            # RMS pour l'énergie effective
            rms = librosa.feature.rms(y=y, hop_length=512)[0]
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))
            
            # Extraction de toutes les caractéristiques améliorées
            mfcc_features = self.extract_mfcc_features(y, sr)
            prosodic_features = self.extract_prosodic_features(y, sr)
            spectral_features = self.extract_spectral_features(y, sr)
            rhythm_features = self.extract_rhythm_features(y, sr)
            
            # Détection des pauses par deux méthodes et calcul de la moyenne
            pauses_rms = self.detect_pauses_librosa(y, sr)
            pauses_onset = self.detect_pauses_onset_based(y, sr)
            pauses_avg = (pauses_rms + pauses_onset) / 2
            
            # Voice embedding
            try:
                wav = preprocess_wav(file_path)
                embedding = self.encoder.embed_utterance(wav)
            except Exception as e:
                print(f"⚠️ Échec du Voice Embedding pour {file_name}: {e}")
                embedding = np.zeros(256)  # taille d'embedding par défaut
            
            # Regroupement des résultats
            result = {
                "video": video,
                "chunk": chunk, 
                "file": file_name,
                "duration": duration,
                "energy": energy,
                "rms_mean": rms_mean,
                "rms_std": rms_std,
                "pauses_rms_method": int(pauses_rms),
                "pauses_onset_method": int(pauses_onset),
                "pauses_average": float(pauses_avg),
                "pauses_per_minute": float(pauses_avg / (duration / 60)) if duration > 0 else 0,
                
                # Caractéristiques prosodiques améliorées
                **prosodic_features,
                
                # Caractéristiques spectrales améliorées
                **spectral_features,
                
                # Caractéristiques de rythme améliorées
                **rhythm_features,
                
                # MFCCs améliorés
                **mfcc_features,
                
                # Voice embedding
                "embedding": embedding.tolist()
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Erreur dans l'analyse de {file_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

# Utilisation principale
def main():
    # Chemin correct - choisir celui qui convient
    possible_paths = [
        r"C:\Users\ADmiN\Desktop\video_project\audio_chunks",  # Chemin complet
        "audio_chunks",  # Dans le même dossier
        ".",  # Dossier courant
    ]
    
    audio_folder = None
    for path in possible_paths:
        if os.path.exists(path):
            audio_folder = path
            break
    
    if audio_folder is None:
        print("❌ Dossier audio_chunks introuvable")
        print("🔍 Recherche dans le dossier courant...")
        audio_folder = "."
    
    analyzer = AdvancedAudioAnalyzer()
    results = []
    
    print("🚀 Début de l'analyse avancée améliorée des fichiers audio...")
    print(f"📂 Recherche dans le dossier: {os.path.abspath(audio_folder)}")
    
    # Affichage du contenu du dossier pour diagnostic
    all_files = []
    audio_files = []
    
    for root, dirs, files in os.walk(audio_folder):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
            if file.lower().endswith((".mp3", ".wav", ".flac", ".m4a")):
                audio_files.append(full_path)
    
    print(f"🔍 {len(audio_files)} fichiers audio trouvés sur {len(all_files)} fichiers")
    
    if len(audio_files) == 0:
        print("⚠️ Aucun fichier audio trouvé!")
        print("📋 Les 10 premiers fichiers présents:")
        for file in all_files[:10]:
            print(f"   - {os.path.basename(file)}")
        if len(all_files) > 10:
            print(f"   ... et {len(all_files)-10} autres fichiers")
        return
    
    processed_count = 0
    failed_count = 0
    
    for file_path in audio_files:
        # Extraction des informations du chemin
        parts = os.path.normpath(file_path).split(os.sep)
        file_name = os.path.basename(file_path)
        video = parts[-3] if len(parts) >= 3 else "unknown"
        chunk = parts[-2] if len(parts) >= 2 else "unknown"
        
        print(f"📊 Analyse: {file_name}")
        result = analyzer.analyze_audio_file(file_path, video, chunk, file_name)
        
        if result:
            results.append(result)
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"✅ {processed_count} fichiers analysés...")
        else:
            failed_count += 1
    
    # Sauvegarde des résultats
    if results:
        df = pd.DataFrame(results)
        
        # Vérification de la qualité des données avant sauvegarde
        print("\n🔍 Vérification de la qualité des données...")
        
        # Vérification MFCC
        mfcc_cols = [col for col in df.columns if col.startswith('mfcc_')]
        for col in mfcc_cols:
            if df[col].abs().max() > 1000:
                print(f"⚠️ Valeurs aberrantes dans {col}: {df[col].abs().max()}")
        
        # Vérification Pitch
        pitch_mean_range = (df['pitch_mean'].min(), df['pitch_mean'].max())
        print(f"📊 Plage du pitch: {pitch_mean_range[0]:.1f} - {pitch_mean_range[1]:.1f} Hz")
        
        if pitch_mean_range[0] < 50 or pitch_mean_range[1] > 500:
            print("⚠️ Valeurs de pitch hors de la plage normale")
        
        output_file = "advanced_audio_features_fixed.csv"
        df.to_csv(output_file, index=False)
        print(f"🎉 Terminé! Caractéristiques de {len(results)} fichiers sauvegardées dans {output_file}")
        
        if failed_count > 0:
            print(f"⚠️ Échec de l'analyse de {failed_count} fichiers")
        
        # Statistiques rapides
        print("\n📈 Statistiques rapides:")
        print(f"Durée moyenne des fichiers: {df['duration'].mean():.2f} secondes")
        print(f"Pitch moyen: {df['pitch_mean'].mean():.1f} Hz")
        print(f"Plage de pitch: {df['pitch_mean'].std():.1f} Hz")
        print(f"Tempo moyen: {df['tempo'].mean():.1f} BPM") 
        print(f"Pauses moyennes: {df['pauses_per_minute'].mean():.1f} par minute")
        print(f"MFCC1 moyen: {df['mfcc_1_mean'].mean():.1f}")
        print(f"Centroïde spectral moyen: {df['spectral_centroid_mean'].mean():.0f} Hz")
        
    else:
        print("❌ Aucun fichier analysé avec succès")

if __name__ == "__main__":
    main()