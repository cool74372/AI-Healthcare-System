# import tensorflow as tf
# import numpy as np
# import cv2
# import shutil

import tensorflow as tf
import numpy as np
import cv2
import shutil


def make_gradcam_heatmap(img_array, model):

    img_tensor = tf.cast(img_array, tf.float32)

    try:
        # Detect model type
        layer_names = [l.name for l in model.layers]
        is_mobilenet = any("Conv_1" in name or "mobilenetv2" in name.lower()
                           for name in layer_names)

        print(f"[Grad-CAM] Detected as: {'MobileNetV2' if is_mobilenet else 'CNN'}")

        # Select layer
        last_conv_layer = None

        if is_mobilenet:
            for layer in model.layers:
                if layer.name == "Conv_1":
                    last_conv_layer = layer
                    break

        # fallback (important for pneumonia CNN or edge cases)
        if last_conv_layer is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer
                    break

        if last_conv_layer is None:
            print("[Grad-CAM] No conv layer found")
            return None

        print(f"[Grad-CAM] Using layer: {last_conv_layer.name}")

        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            tape.watch(conv_outputs)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            print("[Grad-CAM] Gradients None")
            grads = tf.ones_like(conv_outputs)

        # 🔥 FIX 1: remove abs() → better localization
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_out = conv_outputs[0]

        # 🔥 FIX 2: stable Grad-CAM computation
        heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)

        heatmap = tf.nn.relu(heatmap)

        # Normalize
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

        # 🔥 FIX 3: remove scattered weak regions (Sharper threshold)
        heatmap = tf.where(heatmap < 0.45, 0.0, heatmap)

        print("[Grad-CAM] Heatmap generated successfully")
        return heatmap.numpy()

    except Exception as e:
        print(f"[Grad-CAM] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def overlay_heatmap(heatmap, image_path, output_path, alpha=0.4):

    img = cv2.imread(image_path)
    if img is None:
        print("Image not found:", image_path)
        return None

    if heatmap is None:
        shutil.copy(image_path, output_path)
        return output_path

    h, w = img.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))

    # 🔥 FIX 4: smooth + sharpen
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (7, 7), 0)
    heatmap_resized = np.power(heatmap_resized, 2.5)

    heatmap_resized = heatmap_resized / (np.max(heatmap_resized) + 1e-8)

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    mask = cv2.GaussianBlur(heatmap_resized, (11, 11), 0)
    mask = np.stack([mask, mask, mask], axis=-1)

    img_f = img.astype(np.float32)
    heat_f = heatmap_color.astype(np.float32)

    result = img_f * (1 - mask * alpha) + heat_f * (mask * alpha)
    result = np.clip(result, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, result)
    print(f"[Grad-CAM] Overlay saved: {output_path}")

    return output_path

# # ─────────────────────────────────────────
# # FIND LAST CONV LAYER (WORKS FOR BOTH MODELS)
# # ─────────────────────────────────────────
# def get_last_conv_layer(model):
#     for layer in reversed(model.layers):

#         # Case 1: Sequential model with submodel (MobileNetV2)
#         if hasattr(layer, 'layers'):
#             for sub_layer in reversed(layer.layers):
#                 if isinstance(sub_layer, tf.keras.layers.Conv2D):
#                     return sub_layer.name, layer.name  # (conv_name, parent_name)

#         # Case 2: Functional / flat model
#         if isinstance(layer, tf.keras.layers.Conv2D):
#             return layer.name, None

#     raise ValueError("No Conv2D layer found.")


# # ─────────────────────────────────────────
# # GET FINAL DENSE LAYER
# # ─────────────────────────────────────────
# def get_final_dense(model):
#     dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
#     final_dense = dense_layers[-1]
#     return final_dense, final_dense.name


# # ─────────────────────────────────────────
# # BUILD FEATURE MODEL (FIXED VERSION 🔥)
# # ─────────────────────────────────────────
# def build_feature_model(model, last_conv_name, parent_name, final_dense_name):

#     # Find index of final dense
#     final_dense_idx = None
#     for i, layer in enumerate(model.layers):
#         if layer.name == final_dense_name:
#             final_dense_idx = i
#             break

#     pre_dense_output = model.layers[final_dense_idx - 1].output

#     # ── CASE 1: Sequential (TB MODEL) ─────────────────
#     if parent_name is not None:
#         sub_model = model.get_layer(parent_name)

#         # 🔥 FIX: Proper graph connection
#         conv_output = sub_model.get_layer(last_conv_name).output

#         conv_model = tf.keras.models.Model(
#             inputs=model.input,
#             outputs=conv_output
#         )

#         feature_model = tf.keras.models.Model(
#             inputs=model.input,
#             outputs=[conv_model.output, pre_dense_output]
#         )

#     # ── CASE 2: Functional (PNEUMONIA MODEL) ─────────
#     else:
#         feature_model = tf.keras.models.Model(
#             inputs=model.input,
#             outputs=[
#                 model.get_layer(last_conv_name).output,
#                 pre_dense_output
#             ]
#         )

#     return feature_model


# # ─────────────────────────────────────────
# # GRAD-CAM CORE
# # ─────────────────────────────────────────
# def make_gradcam_heatmapx(img_array, model, pred_index=None):

#     img_tensor = tf.cast(img_array, tf.float32)

#     # Prediction
#     raw_pred = float(model(img_tensor, training=False)[0][0])
#     print(f"[Grad-CAM] Prediction: {raw_pred:.4f}")

#     # Skip low-confidence normal
#     if raw_pred < 0.5:
#         print("[Grad-CAM] Normal / low confidence — skipping.")
#         return None

#     # Get layers
#     last_conv_name, parent_name = get_last_conv_layer(model)
#     final_dense, final_dense_name = get_final_dense(model)

#     print(f"[Grad-CAM] Last conv: {last_conv_name} | Parent: {parent_name}")

#     # Build feature model
#     feature_model = build_feature_model(
#         model, last_conv_name, parent_name, final_dense_name
#     )

#     # Extract weights
#     W, b = final_dense.get_weights()

#     # Gradient calculation
#     with tf.GradientTape() as tape:
#         conv_outputs, features = feature_model(img_tensor)
#         tape.watch(conv_outputs)

#         # Manual logit (IMPORTANT FIX)
#         logit = tf.matmul(features, tf.constant(W)) + tf.constant(b)
#         loss = logit[0][0]

#     grads = tape.gradient(loss, conv_outputs)

#     if grads is None:
#         print("[Grad-CAM] ERROR: gradients None")
#         return None

#     # Compute heatmap
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_out = conv_outputs[0]

#     heatmap = conv_out @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)

#     heatmap = tf.maximum(heatmap, 0)

#     max_val = tf.reduce_max(heatmap)
#     if max_val == 0:
#         return None

#     heatmap /= (max_val + 1e-8)

#     print("[Grad-CAM] ✅ Heatmap generated")
#     return heatmap.numpy()


# # ─────────────────────────────────────────
# # OVERLAY FUNCTION
# # ─────────────────────────────────────────
# def overlay_heatmapx(heatmap, image_path, output_path, alpha=0.45):

#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(image_path)

#     if heatmap is None:
#         shutil.copy(image_path, output_path)
#         return output_path

#     h, w = img.shape[:2]

#     heatmap = cv2.resize(heatmap, (w, h))
#     heatmap = heatmap / (np.max(heatmap) + 1e-8)

#     heatmap_uint8 = np.uint8(255 * heatmap)
#     heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

#     mask = cv2.GaussianBlur(heatmap.astype(np.float32), (11, 11), 0)
#     mask = np.stack([mask, mask, mask], axis=-1)

#     img = img.astype(np.float32)
#     heatmap_color = heatmap_color.astype(np.float32)

#     result = img * (1 - mask * alpha) + heatmap_color * (mask * alpha)
#     result = np.clip(result, 0, 255).astype(np.uint8)

#     cv2.imwrite(output_path, result)
#     print(f"[Grad-CAM] Saved: {output_path}")

#     return output_path