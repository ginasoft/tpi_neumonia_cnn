import argparse, json, os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

def load_label_map(classes_json_path):
    if classes_json_path and os.path.exists(classes_json_path):
        with open(classes_json_path, "r", encoding="utf-8") as f:
            class_indices = json.load(f)  # e.g. {"NORMAL":0,"PNEUMONIA":1}
        inv = {v:k for k,v in class_indices.items()}
        return inv
    # fallback razonable si no hay json
    return {0: "NORMAL", 1: "PNEUMONIA"}

def main(args):
    model = load_model(args.model)
    # Detectar tamaño de entrada del modelo
    _, H, W, C = model.input_shape
    assert C == 3, "El modelo espera 3 canales (RGB)."

    # Mapa de clases
    inv_map = load_label_map(args.classes)

    # Cargar imagen
    img = load_img(args.image, target_size=(H, W))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)  # (1,H,W,3)

    # Predicción
    prob = float(model.predict(x, verbose=0).ravel()[0])  # prob de clase 1
    pred = 1 if prob >= args.threshold else 0
    label = inv_map.get(pred, str(pred))

    print(f"Imagen: {args.image}")
    print(f"Prob(PNEUMONIA=1) = {prob:.4f}")
    print(f"Umbral = {args.threshold:.2f}")
    print(f"Predicción = {label}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ruta al modelo .keras (e.g., outputs/best_cnn.keras)")
    ap.add_argument("--image", required=True, help="Ruta a la imagen a evaluar")
    ap.add_argument("--classes", default="outputs/class_indices.json", help="JSON con class_indices")
    ap.add_argument("--threshold", type=float, default=0.5, help="Umbral de decisión")
    args = ap.parse_args()
    main(args)
