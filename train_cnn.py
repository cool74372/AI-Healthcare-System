import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "data", "chest_xray", "train")
VAL_DIR   = os.path.join(BASE_DIR, "data", "chest_xray", "val")

print("Train Dir:", TRAIN_DIR)
print("Val Dir:", VAL_DIR)

# ─────────────────────────────────────────
# DATA GENERATORS
# ─────────────────────────────────────────

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ─────────────────────────────────────────
# CLASS WEIGHTS (IMPORTANT)
# ─────────────────────────────────────────

normal_count = np.sum(train_data.classes == 0)
pneumonia_count = np.sum(train_data.classes == 1)

total = normal_count + pneumonia_count

class_weight = {
    0: total / (2 * normal_count),
    1: total / (2 * pneumonia_count)
}

print(f"\nClass Weights → NORMAL: {class_weight[0]:.2f}, PNEUMONIA: {class_weight[1]:.2f}")

# ─────────────────────────────────────────
# MODEL (MobileNetV2)
# ─────────────────────────────────────────

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# Custom head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.summary()

# ─────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────

os.makedirs("models", exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='models/pneumonia_mobilenet_best.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# ─────────────────────────────────────────
# TRAIN PHASE 1
# ─────────────────────────────────────────

print("\n🔵 Phase 1: Training classifier head...")

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weight,
    callbacks=callbacks
)

# ─────────────────────────────────────────
# TRAIN PHASE 2 (FINE-TUNING)
# ─────────────────────────────────────────

print("\n🟠 Phase 2: Fine-tuning last layers...")

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weight,
    callbacks=callbacks
)

# ─────────────────────────────────────────
# SAVE FINAL MODEL
# ─────────────────────────────────────────

model.save("models/pneumonia_model.h5")

print("\n✅ Training complete!")
print("📊 Best model: models/pneumonia_mobilenet_best.keras")
print("💾 Final model: models/pneumonia_model.h5")