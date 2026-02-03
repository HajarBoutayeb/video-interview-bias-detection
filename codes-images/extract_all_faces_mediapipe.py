import os
import cv2
from PIL import Image
from tqdm import tqdm
import mediapipe as mp

# ======= Paramètres modifiables =======
# Dossier source des images (racine contenant les clips et images)
INPUT_DIR = r"C:\Users\ADmiN\Desktop\video_project\grouped_images\video1"

# Dossier de sortie pour les visages (contiendra des sous-dossiers pour chaque clip)
OUTPUT_DIR = r"C:\Users\ADmiN\Desktop\video_project\faces\video1"

# Taille de sortie
OUT_SIZE = (256, 256)

# Pourcentage de marge autour du visage avant découpage
MARGIN = 0.25

# Formats d'images acceptés
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# =====================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_fd = mp.solutions.face_detection

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def add_margin(box, img_w, img_h, margin):
    x, y, w, h = box
    mx = int(w * margin)
    my = int(h * margin)
    x1 = clamp(x - mx, 0, img_w - 1)
    y1 = clamp(y - my, 0, img_h - 1)
    x2 = clamp(x + w + mx, 0, img_w - 1)
    y2 = clamp(y + h + my, 0, img_h - 1)
    return x1, y1, x2 - x1, y2 - y1

def detect_faces_all(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_bgr.shape[:2]
    faces = []

    for model_sel in (0, 1):
        with mp_fd.FaceDetection(model_selection=model_sel, min_detection_confidence=0.4) as fd:
            res = fd.process(img_rgb)
            if res.detections:
                for det in res.detections:
                    rel = det.location_data.relative_bounding_box
                    x = int(rel.xmin * W)
                    y = int(rel.ymin * H)
                    w = int(rel.width * W)
                    h = int(rel.height * H)
                    x = clamp(x, 0, W - 1)
                    y = clamp(y, 0, H - 1)
                    w = max(1, min(w, W - x))
                    h = max(1, min(h, H - y))
                    faces.append((x, y, w, h))
    return faces

def list_images(root):
    paths = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(EXTS):
                paths.append(os.path.join(r, f))
    return paths

def tidy_name(path):
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 2:
        clip = parts[-2]  # video9_clip_001
    else:
        clip = "clip"
    base = os.path.splitext(os.path.basename(path))[0]
    return clip, base

def save_face(crop_bgr, clip, base, idx):
    # Sauvegarde du visage dans le dossier du clip correspondant
    clip_dir = os.path.join(OUTPUT_DIR, clip)
    os.makedirs(clip_dir, exist_ok=True)

    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)).resize(OUT_SIZE)
    out_name = f"{base}_face{idx:02d}.jpg"
    out_path = os.path.join(clip_dir, out_name)
    pil.save(out_path, quality=95)
    return out_path

def main():
    all_imgs = list_images(INPUT_DIR)
    if not all_imgs:
        print("⚠️ Aucune image trouvée dans le chemin:", INPUT_DIR)
        return

    print(f"🔎 Traitement de {len(all_imgs)} images depuis '{INPUT_DIR}' et extraction des visages vers '{OUTPUT_DIR}'")
    saved_total = 0
    noface_count = 0

    for img_path in tqdm(all_imgs, desc="Extracting faces"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]
        faces = detect_faces_all(img)

        if not faces:
            noface_count += 1
            continue

        clip, base = tidy_name(img_path)

        for i, (x, y, w, h) in enumerate(faces, start=1):
            x, y, w, h = add_margin((x, y, w, h), W, H, MARGIN)
            crop = img[y:y+h, x:x+w]
            save_face(crop, clip, base, i)
            saved_total += 1

    print(f"✅ Opération terminée! {saved_total} visage(s) sauvegardé(s). Images sans visages: {noface_count}.")
    print(f"📂 Dossier final: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()