from ultralytics import YOLO
import torch
import os
import face_recognition as fr
import numpy as np
import cv2
from omegaconf import OmegaConf
import yaml
import sys


# Ajouter le chemin vers le repo LaMa
sys.path.append("./lama/")

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device


class FastLaMaInpainter:
    """
    Classe optimisée pour l'inpainting en temps réel avec LaMa
    Utilise des optimisations pour améliorer les performances
    """

    def __init__(self, model_path, checkpoint="best.ckpt", max_resolution=512, device="cuda"):
        """
        Initialise le modèle LaMa optimisé pour temps réel

        Args:
            model_path: Chemin vers le dossier du modèle
            checkpoint: Nom du fichier checkpoint
            max_resolution: Résolution maximale pour l'inférence (plus petit = plus rapide)
        """
        self.device = device
        self.max_resolution = max_resolution

        print(f"🚀 Initializing Fast LaMa Inpainter")
        print(f"   Device: {self.device}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   GPU: {gpu_name}")
        elif torch.xpu.is_available():
            gpu_name = torch.xpu.get_device_name(0)
            print(f"   GPU: {gpu_name}")
        elif torch.backends.mps.is_available():
            print(f"   Using Apple Silicon GPU")
        else:
            print("   Using CPU")

        print(f"   Max resolution: {max_resolution}x{max_resolution}")
        # Charger la config
        config_path = os.path.join(model_path, "config.yaml")
        with open(config_path, "r") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"

        # Charger le modèle
        checkpoint_path = os.path.join(model_path, "models", checkpoint)
        print(f"   Loading from: {os.path.basename(model_path)}")

        self.model = load_checkpoint(
            train_config, checkpoint_path, strict=False, map_location=self.device
        )
        self.model.freeze()
        self.model.eval()
        self.model.to(self.device)

        # Optimisations pour temps réel
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

        print("✅ Model loaded and optimized for real-time!")

    def inpaint(self, image, mask):
        """
        Effectue l'inpainting optimisé pour temps réel
        """
        original_shape = image.shape[:2]

        # Downscale si nécessaire pour vitesse
        # IMPORTANT: Arrondir aux multiples de 8 pour éviter les erreurs cuFFT
        if max(original_shape) > self.max_resolution:
            scale = self.max_resolution / max(original_shape)
            new_h = int(image.shape[0] * scale)
            new_w = int(image.shape[1] * scale)
            # Arrondir au multiple de 8 le plus proche
            new_h = ((new_h + 7) // 8) * 8
            new_w = ((new_w + 7) // 8) * 8
            new_size = (new_w, new_h)
            image_resized = cv2.resize(image, new_size)
            mask_resized = cv2.resize(mask, new_size)
        else:
            image_resized = image
            mask_resized = mask

        # Convertir BGR -> RGB et normaliser
        image_rgb = (
            cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        )
        mask_norm = (mask_resized > 127).astype(np.float32)

        # Pad pour être multiple de 8
        h, w = image_rgb.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8

        if pad_h > 0 or pad_w > 0:
            image_rgb = np.pad(
                image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
            )
            mask_norm = np.pad(mask_norm, ((0, pad_h), (0, pad_w)), mode="reflect")

        # Convertir en tenseurs
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_norm).unsqueeze(0).unsqueeze(0)

        # Préparer le batch
        batch = {"image": image_tensor, "mask": mask_tensor}

        # Inférence
        # NOTE: Désactiver FP16 pour éviter les erreurs cuFFT avec dimensions non-puissance-de-2
        with torch.no_grad():
            batch = {k: v.to(self.device) for k, v in batch.items()}

            batch["mask"] = (batch["mask"] > 0) * 1
            result = self.model(batch)
            inpainted = result["inpainted"][0].permute(1, 2, 0).cpu().numpy()

        # Retirer le padding
        if pad_h > 0 or pad_w > 0:
            inpainted = inpainted[:h, :w]

        # Upscale si nécessaire
        if max(original_shape) > self.max_resolution:
            inpainted = cv2.resize(inpainted, (original_shape[1], original_shape[0]))

        # Convertir en uint8 et BGR
        inpainted = np.clip(inpainted * 255, 0, 255).astype(np.uint8)
        inpainted_bgr = cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR)

        return inpainted_bgr


def get_human_masks(img_path, model, device="cpu"):
    results = model.predict(
        source=img_path, conf=0.25, classes=[0], verbose=False, half=True, device=device
    )
    masks = results[0].masks
    return masks


def get_face_landmarks(img_path):
    image = img_path
    if isinstance(img_path, str):
        image = fr.load_image_file(img_path)

    face_locations = fr.face_locations(image)
    face_landmarks_list = fr.face_landmarks(image, face_locations)
    return face_locations, face_landmarks_list


def draw_face_landmarks(image, face_landmarks_list, face=False):
    if face_landmarks_list is None:
        return image
    if not isinstance(face_landmarks_list, list):
        face_landmarks_list = [face_landmarks_list]

    for coord in face_landmarks_list:
        if coord is None:
            continue
        for _, points in coord.items():
            # Ignorer les clés qui ne sont pas des points de repère
            if not isinstance(points, list):
                continue
            for point in points:
                if not isinstance(point, tuple):
                    continue
                cv2.circle(image, point, 2, (0, 255, 0), -1)
        if "face_mask" in coord and face:
            colored_mask = np.zeros_like(image)
            colored_mask[coord["face_mask"] == 255] = [0, 0, 255]  # Rouge pour la face
            alpha = 0.5  # Transparence
            image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
        if "mask" in coord:
            colored_mask = np.zeros_like(image)
            colored_mask[coord["mask"] != 0] = [255, 0, 0]  # Bleu pour le corps
            alpha = 0.3  # Transparence
            image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    return image


def basic_impaint(image, informations, bg_image=None):
    """
    Inpaint the areas defined by the masks in the image.
    If bg_image is provided, use it as the background for inpainting.
    """
    if informations is None:
        return image
    if not isinstance(informations, list):
        informations = [informations]

    inpainted_image = image.copy()
    for info in informations:
        if "mask" not in info:
            continue
        mask_uint8 = (info["mask"].data * 255).numpy().astype(np.uint8)
        if bg_image is not None:
            # Resize bg_image to match the size of the original image
            bg_resized = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
            inpainted_image[mask_uint8 != 0] = bg_resized[mask_uint8 != 0]
        else:
            inpainted_image = cv2.inpaint(
                inpainted_image, mask_uint8, 3, cv2.INPAINT_TELEA
            )

    return inpainted_image


def LaMa_inpaint(
    image: np.ndarray, informations: list[dict], lama_model: FastLaMaInpainter
):
    """
    Inpaint the areas defined by the masks in the image using LaMa model.
    """
    if informations is None:
        return image
    if not isinstance(informations, list):
        informations = [informations]

    result = image.copy()
    for info in informations:
        if "mask" not in info:
            continue
        mask_uint8 = (info["mask"].data * 255).numpy().astype(np.uint8)
        inpainted = lama_model.inpaint(image, mask_uint8)
        result[mask_uint8 != 0] = inpainted[mask_uint8 != 0]
    return result


def associate_faces_with_masks(face_landmarks_list, masks):
    """
    Associe chaque visage détecté avec son masque humain correspondant.
    Dans face_landmarks_list, pour chaque visage, on calcule le point moyen des points de repère du visage.
    Puis on ajoute le masque humain correspondant à ce visage en fonction de la position du point moyen.
    new keys added to each coord dict: 'avg_point', 'mask', 'face_mask'
    """
    for coord in face_landmarks_list:
        avg_x, avg_y = 0, 0
        for _, points in coord.items():
            for point in points:
                avg_x += point[0]
                avg_y += point[1]
        num_points = sum(len(points) for points in coord.values())
        if num_points > 0:
            avg_x //= num_points
            avg_y //= num_points

        # Associate each face with it's mask based on: mask[avg_y, avg_x] != 0
        if masks is None:
            continue
        for mask in masks.data:
            if mask[avg_y, avg_x] != 0:
                coord["avg_point"] = (avg_x, avg_y)
                coord["mask"] = mask
                # Créer un masque de la face, on prend le mask original mais uniquement au dessus du menton.
                face_mask = np.zeros_like(mask, dtype=np.uint8)
                chin_points = coord.get("chin", [])
                if chin_points:
                    chin_y = max(point[1] for point in chin_points)
                    face_mask[:chin_y, :][mask[:chin_y, :] != 0] = 255
                coord["face_mask"] = face_mask

                break
    return face_landmarks_list


def extract_information(
    image, yolo_model, human_to_recognize_enc=None, device="cpu", tolerance=0.5
):
    """
    2 fonctions:
    - si une image d'une personne à reconnaître est fournie, comparer les visages détectés avec cette personne, ne retourner que son mask + points de repère
    - sinon, retourner tous les visages détectés avec leurs masks + points de repère

    Returns:
    [dict_keys(['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip',
        'left_eye', 'right_eye', 'top_lip', 'bottom_lip', 'avg_point', 'mask', 'face_mask'])]
    """
    if isinstance(image, str):
        image = fr.load_image_file(image)

    masks = get_human_masks(image, yolo_model, device=device)  # YOLO
    face_locations, face_landmarks_list = get_face_landmarks(image)

    # Si aucun visage n'est détecté, retourner None ou liste vide
    if not face_locations or not face_landmarks_list:
        return None

    face_landmarks_list = associate_faces_with_masks(face_landmarks_list, masks)

    if human_to_recognize_enc is not None and len(human_to_recognize_enc) > 0:
        image_faces_enc = fr.face_encodings(image, face_locations)

        # Parcourir chaque visage détecté (avec son INDEX correct)
        for i, image_face_enc in enumerate(image_faces_enc):
            # Comparer avec la personne à reconnaître
            correspondances = fr.compare_faces(
                human_to_recognize_enc, image_face_enc, tolerance=tolerance
            )

            # Si ce visage correspond à la personne recherchée
            if True in correspondances:
                # Retourner le visage à l'INDEX i (pas l'index de correspondances!)
                return face_landmarks_list[i] if i < len(face_landmarks_list) else None

        # Aucune correspondance trouvée
        return None
    else:
        return face_landmarks_list


if __name__ == "__main__":
    model_path = "models/yolo11n-seg.pt"
    openvino = torch.xpu.is_available()
    ov_device = None
    if openvino:
        openvino_model_path = "models/yolo11n-seg_openvino_model_int8"

        seg_model = YOLO(openvino_model_path, task="segment", verbose=False)
        ov_device = "intel:gpu"
    else:
        seg_model = YOLO(model_path, task="segment", verbose=False)
        
        
    torch_device = (
        "cuda"
        if torch.cuda.is_available()
        else (
            "xpu"
            if torch.xpu.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    )

    lama_model = FastLaMaInpainter(
        model_path="models/lama-places/big-lama-regular",
        checkpoint="best.ckpt",
        max_resolution=512,
        device=torch_device,
    )

    image_de_qqn = "data/matteo.jpg"
    human_to_recognize_enc = None
    if image_de_qqn is not None and isinstance(image_de_qqn, str):
        human_photo = fr.load_image_file(image_de_qqn)
        human_to_recognize_enc = fr.face_encodings(human_photo, num_jitters=8)

    cap = cv2.VideoCapture(4, cv2.CAP_V4L2)
    rec = cv2.VideoWriter(
        f'output/output_{"ALL" if image_de_qqn is None else image_de_qqn.split("/")[-1]}.mp4',
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (640, 480),
    )

    title = "Real-time Inpainting - Press ESC to Exit"
    cv2.namedWindow(winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la caméra.")
        exit()

    # État de l'inpainting
    inpainting_enabled = True
    inpainting_mode = "lama"  # "lama" ou "basic"

    # Afficher les contrôles au démarrage
    print("\n" + "=" * 60)
    print("🎮 CONTRÔLES")
    print("=" * 60)
    print("  I : Activer/Désactiver l'inpainting")
    print("  L : Mode inpainting LaMa")
    print("  B : Mode inpainting Basic")
    print("  R : Rafraîchir l'image de référence (pour mode Basic)")
    print("  Q / ESC : Quitter")
    print("=" * 60 + "\n")

    for i in range(50):
        ret, first_frame = cap.read()
        if not ret:
            print("Erreur lors de la capture de l'image.")
            exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur lors de la capture de l'image.")
            break

        # Traiter l'image capturée (frame) ici si nécessaire
        informations = extract_information(
            frame, seg_model, human_to_recognize_enc, device=ov_device if openvino else torch_device
        )

        # Ne dessiner que si on a des informations valides
        if informations is not None:
            # Dessiner les points de repère et les masques
            # TODO: remove
            # frame = draw_face_landmarks(frame, informations, face=True)

            # Appliquer l'inpainting selon le mode sélectionné
            if inpainting_enabled:
                if inpainting_mode == "lama":
                    frame = LaMa_inpaint(frame, informations, lama_model=lama_model)
                elif inpainting_mode == "basic":
                    frame = basic_impaint(frame, informations, bg_image=first_frame)
        else:
            cv2.putText(
                frame,
                "No recognized face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # Afficher l'état en bas à gauche
        status_text = f"Inpainting: {'ON' if inpainting_enabled else 'OFF'}"
        if inpainting_enabled:
            status_text += f" ({inpainting_mode.upper()})"

        cv2.putText(
            frame,
            status_text,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0) if inpainting_enabled else (0, 0, 255),
            1,
        )

        cv2.imshow(title, frame)
        rec.write(frame)
        key = cv2.waitKey(1) & 0xFF  # Minimal wait for key processing

        # Gestion des touches
        if key == 27 or key == ord("q"):  # ESC or Q to quit
            break
        elif key == ord("i") or key == ord("I"):
            inpainting_enabled = not inpainting_enabled
            print(f"Inpainting: {'activé' if inpainting_enabled else 'désactivé'}")
        elif key == ord("l") or key == ord("L"):
            if inpainting_enabled:
                inpainting_mode = "lama"
                print("Mode: LaMa inpainting")
        elif key == ord("b") or key == ord("B"):
            if inpainting_enabled:
                inpainting_mode = "basic"
                print("Mode: Basic inpainting")
        elif key == ord("r") or key == ord("R"):
            ret, first_frame = cap.read()
            if ret:
                print("Image de référence rafraîchie")
            else:
                print("Erreur lors de la capture de l'image de référence")

    cap.release()
    cv2.destroyAllWindows()
    rec.release()
