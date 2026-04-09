# import tensorflow as tf
# import numpy as np
# import cv2
# import os


import tensorflow as tf
import numpy as np
import cv2
import shutil


def make_gradcam_heatmap(img_array, model):

    img_tensor = tf.cast(img_array, tf.float32)

    try:
        # ✅ FIX: detect MobileNetV2 by checking layer names, not isinstance
        # TB model (MobileNetV2) has a layer named "Conv_1"
        # CNN model (Pneumonia) does NOT have "Conv_1"
        layer_names = [l.name for l in model.layers]
        is_mobilenet = any("Conv_1" in name or "mobilenetv2" in name.lower() 
                          for name in layer_names)

        print(f"[Grad-CAM TB] Layer names: {layer_names[:5]}...")
        print(f"[Grad-CAM TB] Detected as: {'MobileNetV2 (TB)' if is_mobilenet else 'Sequential CNN'}")

        if is_mobilenet:
            # ✅ TB MODEL: find Conv_1 directly in model layers
            last_conv_layer = None
            for layer in model.layers:
                if layer.name == "Conv_1":
                    last_conv_layer = layer
                    break

            # fallback: find last Conv2D
            if last_conv_layer is None:
                for layer in reversed(model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        last_conv_layer = layer
                        break

        else:
            # ✅ CNN MODEL: find last Conv2D
            last_conv_layer = None
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer
                    break

        if last_conv_layer is None:
            print("[Grad-CAM TB] No conv layer found")
            return None

        print(f"[Grad-CAM TB] Using layer: {last_conv_layer.name}")

        grad_model = tf.keras.models.Model(
            inputs=model.input,             # ✅ .input works for both model types
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            tape.watch(conv_outputs)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            print("[Grad-CAM TB] Gradients None")
            grads = tf.ones_like(conv_outputs)

        pooled_grads = tf.reduce_mean(tf.abs(grads), axis=(0, 1, 2))
        conv_out = conv_outputs[0]

        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.abs(heatmap)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        
        # 🔥 FIX: remove scattered weak regions
        heatmap = tf.where(heatmap < 0.45, 0.0, heatmap)

        print("[Grad-CAM TB] Heatmap generated successfully")
        return heatmap.numpy()

    except Exception as e:
        print(f"[Grad-CAM TB] Error: {e}")
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
    
    # 🔥 FIX: smooth + sharpen
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (7, 7), 0)
    heatmap_resized = np.power(heatmap_resized, 2.5)
    heatmap_resized = heatmap_resized / (np.max(heatmap_resized) + 1e-8)
    
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    mask = cv2.GaussianBlur(heatmap_resized.astype(np.float32), (11, 11), 0)
    mask = np.stack([mask, mask, mask], axis=-1)

    img_f = img.astype(np.float32)
    heat_f = heatmap_color.astype(np.float32)

    result = img_f * (1 - mask * alpha) + heat_f * (mask * alpha)
    result = np.clip(result, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, result)
    print(f"[Grad-CAM TB] Overlay saved: {output_path}")
    return output_path

# # 🔹 Get last conv layer (MobileNetV2 fix included)
# def get_last_conv_layer(model):
#     try:
#         base_model = model.layers[0]  # MobileNetV2
#         return base_model.get_layer("Conv_1"), base_model
#     except:
#         for layer in reversed(model.layers):
#             if isinstance(layer, tf.keras.layers.Conv2D):
#                 return layer, model
#     return None, None


# # 🔹 Generate Grad-CAM
# def make_gradcam_heatmap(img_array, model):

#     img_tensor = tf.cast(img_array, tf.float32)

#     # -------------------------------
#     # 🔥 STEP 1: Detect model type
#     # -------------------------------
#     is_mobilenet = isinstance(model.layers[0], tf.keras.Model)

#     if is_mobilenet:
#         # ✅ TB MODEL (MobileNetV2)
#         base_model = model.layers[0]

#         # last conv layer in MobileNetV2
#         last_conv_layer = base_model.get_layer("Conv_1")

#         grad_model = tf.keras.models.Model(
#             inputs=base_model.input,
#             outputs=[last_conv_layer.output, base_model.output]
#         )

#         with tf.GradientTape() as tape:
#             conv_outputs, features = grad_model(img_tensor)
#             tape.watch(conv_outputs)

#             # pass through classifier head
#             x = features
#             for layer in model.layers[1:]:
#                 x = layer(x)

#             loss = x[:, 0]

#     else:
#         # ✅ CNN MODEL (PNEUMONIA)

#         # 🔥 Get LAST TRUE conv layer (not random one)
#         last_conv_layer = None
#         for layer in reversed(model.layers):
#             if isinstance(layer, tf.keras.layers.Conv2D) and len(layer.output_shape) == 4:
#                 last_conv_layer = layer
#                 break

#         if last_conv_layer is None:
#             print("❌ No valid conv layer found")
#             return None

#         print("Using CNN conv layer:", last_conv_layer.name)

#         grad_model = tf.keras.models.Model(
#             inputs=model.input,
#             outputs=[last_conv_layer.output, model.output]
#         )

#         with tf.GradientTape() as tape:
#             conv_outputs, predictions = grad_model(img_tensor)
#             loss = predictions[:, 0]

#     # -------------------------------
#     # 🔥 STEP 2: Compute gradients
#     # -------------------------------
#     grads = tape.gradient(loss, conv_outputs)

#     if grads is None:
#         print("⚠️ Gradients None → using fallback gradients")
#         grads = tf.ones_like(conv_outputs)

#     # 🔥 IMPORTANT: use absolute gradients (fix weak maps)
#     pooled_grads = tf.reduce_mean(tf.abs(grads), axis=(0, 1, 2))

#     conv_outputs = conv_outputs[0]

#     # -------------------------------
#     # 🔥 STEP 3: Build heatmap
#     # -------------------------------
#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)

#     # 🔥 keep all signal (not just positive)
#     heatmap = tf.abs(heatmap)

#     # -------------------------------
#     # 🔥 STEP 4: Normalize safely
#     # -------------------------------
#     heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

#     print("✅ Grad-CAM generated")

#     return heatmap.numpy()


# # 🔹 Overlay heatmap
# def overlay_heatmap(heatmap, image_path, output_path):

#     img = cv2.imread(image_path)

#     if img is None:
#         print("❌ Image not found:", image_path)
#         return None

#     if heatmap is None:
#         print("⚠️ Using original image as fallback")
#         cv2.imwrite(output_path, img)
#         return output_path

#     h, w = img.shape[:2]

#     heatmap = cv2.resize(heatmap, (w, h))
#     heatmap = np.uint8(255 * heatmap)

#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     superimposed_img = cv2.addWeighted(img, 0.5, heatmap, 0.3, 0)

#     cv2.imwrite(output_path, superimposed_img)

#     print("✅ Grad-CAM saved:", output_path)

#     return output_path