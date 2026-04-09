import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

IMG_SIZE = 224
BATCH_SIZE = 32

# ─────────────────────────────────────────
# DATA
# ─────────────────────────────────────────

train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'data/tb_xray/Train',           # ✅ TB train folder
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_datagen.flow_from_directory(
    'data/tb_xray/Validate',        # ✅ TB validate folder
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ─────────────────────────────────────────
# CLASS WEIGHTS (fixes overconfidence)
# ─────────────────────────────────────────

normal_count = train_data.classes.tolist().count(0)
tb_count = train_data.classes.tolist().count(1)
total = normal_count + tb_count

class_weight = {
    0: total / (2 * normal_count),   # NORMAL
    1: total / (2 * tb_count)        # TUBERCULOSIS
}
print(f"Class weights → NORMAL: {class_weight[0]:.2f}, TB: {class_weight[1]:.2f}")

# ─────────────────────────────────────────
# MODEL (MobileNetV2)
# ─────────────────────────────────────────

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base for Phase 1
for layer in base_model.layers:
    layer.trainable = False

# Custom classifier head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),  # ✅ reduces overconfidence
    metrics=['accuracy']
)

# ─────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────

callbacks_phase1 = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(
        filepath='models/tb_model_best.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
]

callbacks_phase2 = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(
        filepath='models/tb_model_best.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
]

# ─────────────────────────────────────────
# PHASE 1 — Train classifier head only
# ─────────────────────────────────────────

print("\n🔵 Phase 1: Training classifier head (base frozen)...")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weight,      # ✅ fixes overconfidence
    callbacks=callbacks_phase1
)

# ─────────────────────────────────────────
# PHASE 2 — Fine-tune last 30 layers
# ─────────────────────────────────────────

print("\n🟠 Phase 2: Fine-tuning last 30 layers...")

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # ✅ very low LR for fine-tuning
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weight,
    callbacks=callbacks_phase2
)

# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────

model.save('models/tb_model.h5')    # ✅ FIXED: was cnn_model.h5
print("✅ TB Model Trained & Saved!")
print("💾 Best checkpoint also saved at: models/tb_model_best.keras")
print()
print("📊 Tip: If val_accuracy is consistently above 80%, your TB model is ready for the app.")