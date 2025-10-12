import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(args):
    DATA_DIR = args.data_dir
    TEST_DIR = os.path.join(DATA_DIR, "test")
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    IMG_SIZE = (224, 224)
    BATCH = args.batch

    model = load_model(args.model_path)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH,
        class_mode='binary', shuffle=False
    )

    y_true = test_gen.classes
    probs = model.predict(test_gen)
    y_pred = (probs.ravel() >= 0.5).astype(int)

    # Reporte
    target_names = list(test_gen.class_indices.keys())  # e.g., ['NORMAL', 'PNEUMONIA']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print(report)

    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format='d')
    plt.title("Matriz de confusión (test)")
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    fpr, tpr, _ = roc_curve(y_true, probs.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("1 - Especificidad")
    plt.ylabel("Sensibilidad")
    plt.title("ROC (test)")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    prec, rec, _ = precision_recall_curve(y_true, probs.ravel())
    ap = average_precision_score(y_true, probs.ravel())
    plt.figure()
    plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (test)")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "pr_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("Resultados guardados en:", OUT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/chest_xray")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--model-path", type=str, default="outputs/best_cnn.keras")
    args = parser.parse_args()
    main(args)
