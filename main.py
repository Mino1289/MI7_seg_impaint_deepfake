from ultralytics import YOLO
import torch
import face_recognition as fr
import numpy as np
import cv2


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
            frame = draw_face_landmarks(frame, informations, face=True)
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

        cv2.imshow(title, frame)
        rec.write(frame)
        key = cv2.waitKey(1) & 0xFF  # Minimal wait for key processing

        # Gestion des touches
        if key == 27 or key == ord("q"):  # ESC or Q to quit
            break
        
    
    cap.release()
    cv2.destroyAllWindows()
    rec.release()
    
