import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
import tensorflow as tf
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def build_simple_cnn(input_shape=(224,224,3)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def main(args):
    # Seeds (reproducibilidad)
    np.random.seed(42)
    tf.random.set_seed(42)

    DATA_DIR = args.data_dir  # e.g., data/chest_xray
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR   = os.path.join(DATA_DIR, "val")
    TEST_DIR  = os.path.join(DATA_DIR, "test")
    OUT_DIR   = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    IMG_SIZE = (180, 180)
    BATCH = args.batch
    EPOCHS = args.epochs

    # Generadores
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.08,
        horizontal_flip=True
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH,
        class_mode='binary', shuffle=True, seed=42
    )
    val_gen = val_test_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH,
        class_mode='binary', shuffle=False
    )
    test_gen = val_test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH,
        class_mode='binary', shuffle=False
    )

    print("Classes mapping:", train_gen.class_indices)

    import json, os
    with open(os.path.join(OUT_DIR, "class_indices.json"), "w") as f:
    json.dump(train_gen.class_indices, f)

    # class_weight
    y_train_all = []
    for _ in range(len(train_gen)):
        _, yb = next(train_gen)
        y_train_all.extend(yb)
    y_train_all = np.array(y_train_all, dtype=int)
    train_gen.reset()

    classes = np.unique(y_train_all)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_all)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
    print("class_weight:", class_weight_dict)

    # Modelo
    model = build_simple_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.summary()

    # Callbacks
    ckpt_path = os.path.join(OUT_DIR, "best_cnn.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss',
                                           save_best_only=True, mode='min', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                         restore_best_weights=True)
    ]

    # Entrenamiento
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    # Guardar modelo final también
    final_path = os.path.join(OUT_DIR, "final_cnn.keras")
    model.save(final_path)
    print(f"Guardado: {final_path}")

    # Curvas
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure()
    plt.plot(acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, 'acc_curve.png'), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, 'loss_curve.png'), dpi=150)
    plt.close()

    print("Curvas guardadas en:", OUT_DIR)
    print("Listo. Usá evaluate.py para métricas de test.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/chest_xray",
                        help="Ruta a la carpeta chest_xray (contiene train/ val/ test/)")
    parser.add_argument("--out-dir", type=str, default="outputs",
                        help="Carpeta de salida para modelos y gráficos")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    main(args)
