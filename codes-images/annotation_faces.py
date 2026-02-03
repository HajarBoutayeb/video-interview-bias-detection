import os
import pandas as pd
import numpy as np
import cv2
from deepface import DeepFace
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# ===================== PATH =====================
BASE_FOLDER = r"C:\Users\ADmiN\Desktop\video_project\faces"
OUTPUT_CSV = r"C:\Users\ADmiN\Desktop\video_project\faces_annotations_ultra.csv"
OUTPUT_FACE_FOLDER = r"C:\Users\ADmiN\Desktop\video_project\faces_detected"

os.makedirs(OUTPUT_FACE_FOLDER, exist_ok=True)

# ===================== CONFIGURATION AMÉLIORÉE =====================
MIN_FACE_CONFIDENCE = 0.3  # Réduction du seuil pour une meilleure détection
MIN_GENDER_CONFIDENCE = 0.5  # Légère réduction
MIN_EMOTION_CONFIDENCE = 0.4  # Réduction pour obtenir plus de résultats
MIN_RACE_CONFIDENCE = 0.4
MIN_FACE_SIZE = 25  # Réduction de la taille minimale
RESIZE_FACTOR = 0.8  # Meilleur équilibre entre vitesse et précision

# ===================== HELPERS =====================
def get_dominant_fast(dic, min_confidence=0.5):
    """Extraction du résultat le plus fort"""
    if not dic:
        return None, None
    key = max(dic, key=dic.get)
    confidence = dic[key] / 100.0
    return (key, confidence) if confidence >= min_confidence else (None, confidence)

def validate_bbox(x, y, w, h, img_w, img_h):
    """Vérification de la validité des coordonnées du visage"""
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h

def process_single_image_improved(img_path, clip_name, video_name, image_name):
    """Traitement d'une seule image - version améliorée"""
    results = []
    
    try:
        # Chargement de l'image
        img = cv2.imread(img_path)
        if img is None:
            return results
        
        orig_h, orig_w = img.shape[:2]
        
        # Redimensionnement pour le traitement
        new_w, new_h = int(orig_w * RESIZE_FACTOR), int(orig_h * RESIZE_FACTOR)
        img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        start_time = time.time()
        
        # Analyse DeepFace avec gestion des erreurs
        try:
            face_analysis = DeepFace.analyze(
                img_small,
                actions=['age', 'gender', 'emotion', 'race'],
                enforce_detection=False,
                silent=True  # Réduction des messages
            )
        except Exception as e:
            # En cas d'échec de l'analyse, essayer sans race
            try:
                face_analysis = DeepFace.analyze(
                    img_small,
                    actions=['age', 'gender', 'emotion'],
                    enforce_detection=False,
                    silent=True
                )
                # Ajout de race vide
                if isinstance(face_analysis, list):
                    for face in face_analysis:
                        face['race'] = {}
                else:
                    face_analysis['race'] = {}
            except:
                return results
        
        process_time = time.time() - start_time
        
        # Conversion en liste si ce n'est pas le cas
        if not isinstance(face_analysis, list):
            face_analysis = [face_analysis]
            
        for idx, face in enumerate(face_analysis):
            try:
                # Extraction des informations de base
                region = face.get('region', {})
                x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
                
                # Vérification de la validité des coordonnées
                x, y, w, h = validate_bbox(x, y, w, h, new_w, new_h)
                
                # Filtrage des petits visages
                if min(w, h) < MIN_FACE_SIZE:
                    continue
                
                # Extraction des données
                age = face.get('age', None)
                gender, gender_conf = get_dominant_fast(face.get("gender", {}), MIN_GENDER_CONFIDENCE)
                emotion, emotion_conf = get_dominant_fast(face.get("emotion", {}), MIN_EMOTION_CONFIDENCE)
                race, race_conf = get_dominant_fast(face.get("race", {}), MIN_RACE_CONFIDENCE)
                
                # Classification d'âge améliorée
                age_cat = None
                if age:
                    if age < 13: age_cat = "Child"
                    elif age < 20: age_cat = "Teen"
                    elif age < 30: age_cat = "Young Adult"
                    elif age < 45: age_cat = "Adult"
                    elif age < 65: age_cat = "Middle-aged"
                    else: age_cat = "Senior"
                
                # Création d'un identifiant unique pour le visage - correction du problème
                face_id = f"{video_name}_{clip_name}_{os.path.splitext(image_name)[0]}_face{idx+1}"
                
                # Sauvegarde de la région du visage (améliorée)
                face_file = None
                if gender or emotion:  # Sauvegarde même si un seul est valide
                    try:
                        # Conversion des coordonnées vers l'image originale
                        scale = 1.0 / RESIZE_FACTOR
                        x_orig = int(x * scale)
                        y_orig = int(y * scale)
                        w_orig = int(w * scale)
                        h_orig = int(h * scale)
                        
                        # Vérification des limites
                        x_orig, y_orig, w_orig, h_orig = validate_bbox(x_orig, y_orig, w_orig, h_orig, orig_w, orig_h)
                        
                        # Découpage du visage de l'image originale
                        face_crop = img[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig]
                        if face_crop.size > 0 and face_crop.shape[0] > 10 and face_crop.shape[1] > 10:
                            face_file = os.path.join(OUTPUT_FACE_FOLDER, f"{face_id}.jpg")
                            cv2.imwrite(face_file, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    except Exception as save_error:
                        print(f"Erreur lors de la sauvegarde du visage: {save_error}")
                
                # Calcul du score de confiance global
                total_confidence = 0
                confidence_count = 0
                for conf in [gender_conf, emotion_conf, race_conf]:
                    if conf is not None:
                        total_confidence += conf
                        confidence_count += 1
                
                avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
                
                # Ajout des données avec améliorations
                result = {
                    "video": video_name,
                    "clip": clip_name,
                    "image": image_name,
                    "face_id": face_id,
                    "face_index": idx+1,
                    "face_detected": True,
                    "age": age,
                    "age_category": age_cat,
                    "gender": gender,
                    "gender_confidence": round(gender_conf, 3) if gender_conf else None,
                    "emotion": emotion,
                    "emotion_confidence": round(emotion_conf, 3) if emotion_conf else None,
                    "race": race,
                    "race_confidence": round(race_conf, 3) if race_conf else None,
                    "avg_confidence": round(avg_confidence, 3),
                    "bbox_x": int(x_orig) if 'x_orig' in locals() else int(x * (1.0/RESIZE_FACTOR)),
                    "bbox_y": int(y_orig) if 'y_orig' in locals() else int(y * (1.0/RESIZE_FACTOR)),
                    "bbox_w": int(w_orig) if 'w_orig' in locals() else int(w * (1.0/RESIZE_FACTOR)),
                    "bbox_h": int(h_orig) if 'h_orig' in locals() else int(h * (1.0/RESIZE_FACTOR)),
                    "face_thumbnail": face_file,
                    "processing_time": round(process_time, 3),
                    "image_width": orig_w,
                    "image_height": orig_h
                }
                
                results.append(result)
                
            except Exception as face_error:
                print(f"Erreur lors du traitement du visage {idx}: {face_error}")
                continue
                
    except Exception as e:
        print(f"Erreur générale dans {os.path.basename(img_path)}: {str(e)}")
        
    return results

# ===================== TRAITEMENT PRINCIPAL =====================
print("📂 Collection des images...")
image_list = []
supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

for root, dirs, files in os.walk(BASE_FOLDER):
    for file in files:
        if file.lower().endswith(supported_formats):
            img_path = os.path.join(root, file)
            clip_name = os.path.basename(root)
            video_name = os.path.basename(os.path.dirname(root))
            image_list.append((img_path, clip_name, video_name, file))

print(f"📊 {len(image_list)} images trouvées")

if len(image_list) == 0:
    print("❌ Aucune image trouvée! Vérifier le chemin.")
    exit()

# Traitement amélioré
data = []
start_total = time.time()
failed_count = 0

print("🚀 Début du traitement amélioré...")
for i, (img_path, clip_name, video_name, image_name) in enumerate(tqdm(image_list, desc="Traitement des images")):
    results = process_single_image_improved(img_path, clip_name, video_name, image_name)
    
    if results:
        data.extend(results)
    else:
        failed_count += 1
    
    # Affichage de la progression toutes les 25 images
    if (i + 1) % 25 == 0:
        elapsed = time.time() - start_total
        rate = (i + 1) / elapsed
        eta = (len(image_list) - (i + 1)) / rate if rate > 0 else 0
        success_rate = ((i + 1 - failed_count) / (i + 1)) * 100
        print(f"⏱️  Traité: {i+1}/{len(image_list)} | Réussi: {success_rate:.1f}% | Vitesse: {rate:.1f} img/s | Restant: {eta/60:.1f} min")

total_time = time.time() - start_total

# ===================== RÉSULTATS ET STATISTIQUES AMÉLIORÉES =====================
if data:
    df = pd.DataFrame(data)
    
    # Ajout de statistiques supplémentaires
    df['face_area'] = df['bbox_w'] * df['bbox_h']
    df['face_ratio'] = df['bbox_w'] / df['bbox_h']
    
    # Sauvegarde du CSV avec encodage amélioré
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    
    print(f"\n✅ Terminé avec succès!")
    print(f"⏱️  Temps total: {total_time/60:.1f} minutes")
    print(f"📊 Images traitées: {len(image_list)}")
    print(f"👤 Visages détectés: {len(df)}")
    print(f"❌ Images échouées: {failed_count}")
    print(f"⚡ Vitesse: {len(image_list)/total_time:.1f} images/seconde")
    print(f"🎯 Taux de réussite: {((len(image_list) - failed_count) / len(image_list)) * 100:.1f}%")
    print(f"💾 Fichier CSV: {OUTPUT_CSV}")
    
    # Statistiques de qualité améliorées
    valid_counts = {
        'gender': df['gender'].notna().sum(),
        'emotion': df['emotion'].notna().sum(),
        'race': df['race'].notna().sum(),
        'age': df['age'].notna().sum()
    }
    
    print(f"\n📈 Qualité des données:")
    for attr, count in valid_counts.items():
        percentage = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"- {attr}: {count}/{len(df)} ({percentage:.1f}%)")
    
    # Statistiques de confiance
    print(f"\n🎯 Taux de confiance:")
    confidence_cols = ['gender_confidence', 'emotion_confidence', 'race_confidence', 'avg_confidence']
    for col in confidence_cols:
        if col in df.columns:
            mean_conf = df[col].mean()
            if not pd.isna(mean_conf):
                print(f"  - {col.replace('_', ' ')}: {mean_conf:.3f}")
    
    # Sauvegarde d'un rapport détaillé
    stats_file = OUTPUT_CSV.replace('.csv', '_stats.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"Rapport d'analyse des visages\n")
        f.write(f"==================\n\n")
        f.write(f"Total images: {len(image_list)}\n")
        f.write(f"Visages détectés: {len(df)}\n")
        f.write(f"Taux de réussite: {((len(image_list) - failed_count) / len(image_list)) * 100:.1f}%\n")
        f.write(f"Temps total: {total_time/60:.1f} minutes\n")
    
    print(f"📄 Rapport détaillé: {stats_file}")

else:
    print("❌ Aucun visage valide détecté!")
    print(f"Nombre d'images échouées: {failed_count}/{len(image_list)}")

print(f"\n💡 Conseils pour l'amélioration:")
print(f"- Si les résultats sont lents, réduire RESIZE_FACTOR à 0.6")
print(f"- Si les résultats ne sont pas précis, augmenter RESIZE_FACTOR à 0.9")
print(f"- Pour une meilleure précision, augmenter MIN_FACE_SIZE à 40")
print(f"- Taux de réussite acceptable: > 80%")