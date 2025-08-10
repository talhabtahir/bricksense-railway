# import os
# import uuid
# from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
# from PIL import Image, ImageOps, ExifTags
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from werkzeug.utils import secure_filename
# import matplotlib.cm as cm
# import math

# # CONFIG
# UPLOAD_FOLDER = "static/uploads"
# MODEL_FOLDER = "models"
# ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# app = Flask(__name__)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.secret_key = "replace-this-with-a-secret"  # change for production

# # -------------------------
# # Utility functions
# # -------------------------
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# def save_uploaded_file(file_storage):
#     filename = secure_filename(file_storage.filename)
#     uid = uuid.uuid4().hex[:8]
#     out_name = f"{uid}_{filename}"
#     out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_name)
#     file_storage.save(out_path)
#     return out_path, out_name

# def correct_orientation_pil(image):
#     try:
#         for orientation in ExifTags.TAGS.keys():
#             if ExifTags.TAGS[orientation] == 'Orientation':
#                 break
#         exif = image._getexif()
#         if exif is not None:
#             orientation = exif.get(orientation, 1)
#             if orientation == 3:
#                 image = image.rotate(180, expand=True)
#             elif orientation == 6:
#                 image = image.rotate(270, expand=True)
#             elif orientation == 8:
#                 image = image.rotate(90, expand=True)
#     except Exception:
#         pass
#     return image

# def add_white_border(image_pil, border_size=10):
#     return ImageOps.expand(image_pil, border=border_size, fill=(255,255,255))

# def add_canvas(image_pil, fill_color=(255,255,255)):
#     image_width, image_height = image_pil.size
#     canvas_width = image_width + math.ceil(0.015 * image_width)
#     canvas_height = image_height + math.ceil(0.07 * image_height)
#     canvas = Image.new("RGB", (canvas_width, canvas_height), fill_color)
#     paste_position = ((canvas_width - image_width) // 2, (canvas_height - image_height) // 7)
#     canvas.paste(image_pil, paste_position)
#     return canvas

# # -------------------------
# # APP1: TFLite helpers & load models
# # -------------------------
# def load_tflite_model(path):
#     interpreter = tf.lite.Interpreter(model_path=path)
#     interpreter.allocate_tensors()
#     return interpreter

# def run_tflite_inference(interpreter, inputs):
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     # if model expects multiple inputs, assign in order
#     for i, input_tensor in enumerate(inputs):
#         interpreter.set_tensor(input_details[i]['index'], input_tensor.astype(np.float32))
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     return output_data

# # # Load tflite models (update filenames if needed)
# # try:
# #     strength_model = load_tflite_model(os.path.join(MODEL_FOLDER, "brick_FlexureStrength_Reg_model_epoch50.tflite"))
# #     class_model = load_tflite_model(os.path.join(MODEL_FOLDER, "brick_classification_model trial 2.tflite"))
# #     absorption_model = load_tflite_model(os.path.join(MODEL_FOLDER, "brick_absorption_Model.tflite"))
# # except Exception as e:
# #     # If models missing, set to None and show errors on pages
# #     strength_model = class_model = absorption_model = None
# #     print("Error loading tflite models:", e)
# import traceback  # add this at the top of the file if not already imported

# print("Checking contents of models folder:")
# try:
#     model_files = os.listdir(MODEL_FOLDER)
#     print(model_files)
# except Exception as e:
#     print(f"Error listing model folder '{MODEL_FOLDER}': {e}")

# strength_model = None
# class_model = None
# absorption_model = None

# try:
#     print(f"Loading strength model: {os.path.join(MODEL_FOLDER, 'brick_FlexureStrength_Reg_model_epoch50.tflite')}")
#     strength_model = load_tflite_model(os.path.join(MODEL_FOLDER, "brick_FlexureStrength_Reg_model_epoch50.tflite"))
#     print("Strength model loaded successfully")
# except Exception as e:
#     print("Error loading strength model:")
#     traceback.print_exc()

# try:
#     print(f"Loading class model: {os.path.join(MODEL_FOLDER, 'brick_classification_model trial 2.tflite')}")
#     class_model = load_tflite_model(os.path.join(MODEL_FOLDER, "brick_classification_model trial 2.tflite"))
#     print("Class model loaded successfully")
# except Exception as e:
#     print("Error loading class model:")
#     traceback.print_exc()

# try:
#     print(f"Loading absorption model: {os.path.join(MODEL_FOLDER, 'brick_absorption_Model.tflite')}")
#     absorption_model = load_tflite_model(os.path.join(MODEL_FOLDER, "brick_absorption_Model.tflite"))
#     print("Absorption model loaded successfully")
# except Exception as e:
#     print("Error loading absorption model:")
#     traceback.print_exc()

# # -------------------------
# # APP2: Keras model load
# # -------------------------
# keras_model = None
# try:
#     keras_model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "170kmodelv10_version_cam_1.keras"))
# except Exception as e:
#     print("Error loading keras model:", e)
#     keras_model = None

# # -------------------------
# # App2: import_and_predict (adapted)
# # -------------------------
# def import_and_predict_keras(pil_image, sensitivity=9):
#     """
#     Returns: pred_vec (np.array shape (1, n)), and PIL images:
#     image_with_border, contours_with_border, heatmap_image_pil, contoured_image, overlay_img
#     """
#     try:
#         original_img = np.array(pil_image.convert("RGB"))
#         if original_img.shape[-1] == 4:
#             original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)
#         orig_height, orig_width, _ = original_img.shape
#         max_dimension = max(orig_width, orig_height)
#         contour_thickness = max(2, int(max_dimension / 200))

#         img_resized = cv2.resize(original_img, (224, 224))
#         img_tensor = np.expand_dims(img_resized, axis=0) / 255.0

#         # build custom model to fetch intermediate conv outputs
#         # sensitivity index must be valid for your model
#         custom_model = Model(inputs=keras_model.inputs,
#                              outputs=(keras_model.layers[sensitivity].output, keras_model.layers[-1].output))
#         conv_out, pred_vec = custom_model.predict(img_tensor)
#         conv_out = np.squeeze(conv_out)  # e.g., (28,28,32)

#         pred_idx = int(np.argmax(pred_vec))
#         # Resize conv map to original image size
#         heat_map_resized = cv2.resize(conv_out, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
#         heat_map = np.mean(heat_map_resized, axis=-1)
#         heat_map = np.maximum(heat_map, 0)
#         if heat_map.max() != 0:
#             heat_map = heat_map / heat_map.max()
#         else:
#             heat_map = heat_map

#         threshold = 0.5
#         heat_map_thresh = np.uint8(255 * heat_map)
#         _, thresh_map = cv2.threshold(heat_map_thresh, int(255 * threshold), 255, cv2.THRESH_BINARY)
#         contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         heatmap_colored = np.uint8(255 * cm.jet(heat_map)[:, :, :3])
#         heatmap_image = Image.fromarray(heatmap_colored)

#         contoured_img = original_img.copy()
#         cv2.drawContours(contoured_img, contours, -1, (0, 0, 255), contour_thickness)
#         contoured_image = Image.fromarray(contoured_img)

#         # Overlay heatmap on original
#         heatmap_image_rgba = heatmap_image.convert("RGBA")
#         original_img_pil = Image.fromarray(original_img).convert("RGBA")
#         heatmap_overlay = Image.blend(original_img_pil, heatmap_image_rgba, alpha=0.5)

#         heatmap_overlay_rgb = heatmap_overlay.convert("RGB")
#         heatmap_overlay_rgb_np = np.array(heatmap_overlay_rgb)
#         cv2.drawContours(heatmap_overlay_rgb_np, contours, -1, (0, 0, 0), contour_thickness)
#         overlay_img = Image.fromarray(heatmap_overlay_rgb_np)

#         # Add white borders
#         image_with_border = add_white_border(pil_image, 10)
#         contours_with_border = add_white_border(overlay_img, 10)

#         return pred_vec, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img
#     except Exception as e:
#         print("Error in import_and_predict_keras:", e)
#         return None, None, None, None, None, None

# # -------------------------
# # Routes
# # -------------------------
# @app.route("/")
# def index():
#     return render_template("index.html")

# # Predict properties (app1)
# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     if request.method == "POST":
#         # file upload
#         if 'file' not in request.files:
#             flash("No file part")
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == "":
#             flash("No selected file")
#             return redirect(request.url)
#         if not allowed_file(file.filename):
#             flash("File type not allowed")
#             return redirect(request.url)

#         filepath, filename = save_uploaded_file(file)
#         pil_img = Image.open(filepath).convert("RGB")
#         pil_img = correct_orientation_pil(pil_img)

#         # dry weight (from form)
#         try:
#             dry_weight_grams = float(request.form.get("dry_weight", 2800.0))
#         except:
#             dry_weight_grams = 2800.0

#         # Preprocess image for tflite: BGR + resize + normalize
#         image_np = np.array(pil_img)
#         image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
#         resized_image = cv2.resize(image_bgr, (224,224)) / 255.0
#         img_tensor = np.expand_dims(resized_image.astype(np.float32), axis=0)

#         # Normalize dry weight for model
#         min_val, max_val = 2610, 3144
#         dry_weight_norm = (dry_weight_grams - min_val) / (max_val - min_val)
#         MIN_KN, MAX_KN = 2.13, 12.48

#         if class_model is None or strength_model is None or absorption_model is None:
#             flash("One or more models are missing on the server.")
#             return render_template("predict.html", filename=filename, error="Models missing")

#         try:
#             class_pred = run_tflite_inference(class_model, [img_tensor])
#             class_label = int(np.argmax(class_pred[0]))

#             if class_label == 3:
#                 # not a brick
#                 return render_template("predict.html", filename=filename, not_brick=True)
#             else:
#                 label_1 = 1 if class_label == 0 else 0
#                 label_2 = 1 if class_label == 1 else 0
#                 label_3 = 1 if class_label == 2 else 0
#                 tabular_input = np.array([[label_1, label_2, label_3, dry_weight_norm]], dtype=np.float32)

#                 strength_pred = run_tflite_inference(strength_model, [img_tensor, tabular_input])
#                 strength_norm = float(strength_pred[0][0])
#                 strength_denorm = strength_norm * (MAX_KN - MIN_KN) + MIN_KN

#                 tabular_input_abs = np.array([[dry_weight_grams, class_label + 1]], dtype=np.float32)
#                 absorption_pred = run_tflite_inference(absorption_model, [img_tensor, tabular_input_abs])
#                 absorption_real = float(absorption_pred[0][0])

#                 probs = (class_pred[0] * 100).tolist()
#                 # return page with results
#                 return render_template("predict.html",
#                                        filename=filename,
#                                        strength=round(strength_denorm,2),
#                                        absorption=round(absorption_real,2),
#                                        probs=probs,
#                                        class_idx=class_label)
#         except Exception as e:
#             flash(f"Prediction error: {e}")
#             return render_template("predict.html", filename=filename, error=str(e))

#     # GET
#     return render_template("predict.html")

# # Crack detection (app2)
# @app.route("/cracks", methods=["GET", "POST"])
# def cracks():
#     if request.method == "POST":
#         # file
#         if 'file' not in request.files:
#             flash("No file part")
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == "":
#             flash("No selected file")
#             return redirect(request.url)
#         if not allowed_file(file.filename):
#             flash("File type not allowed")
#             return redirect(request.url)

#         filepath, filename = save_uploaded_file(file)
#         pil_img = Image.open(filepath).convert("RGB")
#         pil_img = correct_orientation_pil(pil_img)

#         # sensitivity from form
#         try:
#             sensitivity = int(request.form.get("sensitivity", 9))
#         except:
#             sensitivity = 9

#         if keras_model is None:
#             flash("Crack detection model is missing.")
#             return render_template("cracks.html", filename=filename, error="Model missing")

#         try:
#             predictions, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img = import_and_predict_keras(pil_img, sensitivity=sensitivity)
#             if predictions is None:
#                 flash("Prediction failed.")
#                 return render_template("cracks.html", filename=filename, error="Prediction failed")

#             predicted_class = int(np.argmax(predictions))
#             probs = (predictions[0] * 100).tolist()

#             # Save generated images to disk and pass filenames to template
#             def save_pil(pil_img_obj, suffix):
#                 uid = uuid.uuid4().hex[:8]
#                 fname = f"{uid}_{suffix}.png"
#                 path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
#                 pil_img_obj.save(path)
#                 return fname

#             fname_border = save_pil(image_with_border, "orig")
#             fname_contours = save_pil(contours_with_border, "contours")
#             fname_heatmap = save_pil(heatmap_image, "heatmap")
#             fname_contoured = save_pil(contoured_image, "contoured")
#             fname_overlay = save_pil(overlay_img, "overlay")

#             return render_template("cracks.html",
#                                    filename=filename,
#                                    probs=probs,
#                                    predicted_class=predicted_class,
#                                    fname_border=fname_border,
#                                    fname_contours=fname_contours,
#                                    fname_heatmap=fname_heatmap,
#                                    fname_contoured=fname_contoured,
#                                    fname_overlay=fname_overlay)
#         except Exception as e:
#             flash(f"Error processing image: {e}")
#             return render_template("cracks.html", filename=filename, error=str(e))

#     # GET
#     return render_template("cracks.html")

# # Static route helper (if needed)
# @app.route("/uploads/<path:filename>")
# def uploaded_file(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
import os
import uuid
import math
import traceback
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from PIL import Image, ImageOps, ExifTags
from tensorflow.keras.models import Model
from werkzeug.utils import secure_filename
import matplotlib.cm as cm

# -------------------------
# CONFIG
# -------------------------
UPLOAD_FOLDER = "static/uploads"
MODEL_FOLDER = "models"
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace-this-with-a-secret"  # change for production

# -------------------------
# Utility functions
# -------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def save_uploaded_file(file_storage):
    filename = secure_filename(file_storage.filename)
    uid = uuid.uuid4().hex[:8]
    out_name = f"{uid}_{filename}"
    out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_name)
    file_storage.save(out_path)
    return out_path, out_name

def correct_orientation_pil(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation, 1)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except Exception:
        pass
    return image

def load_image_fix_orientation(path):
    """Load, fix orientation, convert to RGB, remove EXIF."""
    image = Image.open(path)
    image = correct_orientation_pil(image)
    image = image.convert("RGB")
    image.info.pop('exif', None)
    return image

def resize_if_large(pil_img, max_side=1024):
    """Resize image if any dimension exceeds max_side."""
    w, h = pil_img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil_img

def add_white_border(image_pil, border_size=10):
    return ImageOps.expand(image_pil, border=border_size, fill=(255, 255, 255))

def add_canvas(image_pil, fill_color=(255, 255, 255)):
    image_width, image_height = image_pil.size
    canvas_width = image_width + math.ceil(0.015 * image_width)
    canvas_height = image_height + math.ceil(0.07 * image_height)
    canvas = Image.new("RGB", (canvas_width, canvas_height), fill_color)
    paste_position = ((canvas_width - image_width) // 2, (canvas_height - image_height) // 7)
    canvas.paste(image_pil, paste_position)
    return canvas

# -------------------------
# TFLite helpers
# -------------------------
def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def run_tflite_inference(interpreter, inputs):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    for i, input_tensor in enumerate(inputs):
        interpreter.set_tensor(input_details[i]['index'], input_tensor.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# -------------------------
# Load TFLite models
# -------------------------
strength_model = None
class_model = None
absorption_model = None

try:
    strength_model = load_tflite_model(os.path.join(MODEL_FOLDER, "brick_FlexureStrength_Reg_model_epoch50.tflite"))
except:
    traceback.print_exc()

try:
    class_model = load_tflite_model(os.path.join(MODEL_FOLDER, "brick_classification_model trial 2.tflite"))
except:
    traceback.print_exc()

try:
    absorption_model = load_tflite_model(os.path.join(MODEL_FOLDER, "brick_absorption_Model.tflite"))
except:
    traceback.print_exc()

# -------------------------
# Load Keras model
# -------------------------
keras_model = None
try:
    keras_model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "170kmodelv10_version_cam_1.keras"))
except:
    traceback.print_exc()

# -------------------------
# Crack detection prediction
# -------------------------
def import_and_predict_keras(pil_image, sensitivity=9):
    try:
        original_img = np.array(pil_image.convert("RGB"))
        if original_img.shape[-1] == 4:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)

        orig_height, orig_width, _ = original_img.shape
        contour_thickness = max(2, int(max(orig_width, orig_height) / 200))

        img_resized = cv2.resize(original_img, (224, 224))
        img_tensor = np.expand_dims(img_resized, axis=0) / 255.0

        custom_model = Model(inputs=keras_model.inputs,
                             outputs=(keras_model.layers[sensitivity].output, keras_model.layers[-1].output))
        conv_out, pred_vec = custom_model.predict(img_tensor)
        conv_out = np.squeeze(conv_out)

        pred_idx = int(np.argmax(pred_vec))

        heat_map_resized = cv2.resize(conv_out, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
        heat_map = np.mean(heat_map_resized, axis=-1)
        heat_map = np.maximum(heat_map, 0)
        if heat_map.max() != 0:
            heat_map = heat_map / heat_map.max()

        threshold = 0.5
        heat_map_thresh = np.uint8(255 * heat_map)
        _, thresh_map = cv2.threshold(heat_map_thresh, int(255 * threshold), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        heatmap_colored = np.uint8(255 * cm.jet(heat_map)[:, :, :3])
        heatmap_image = Image.fromarray(heatmap_colored)

        contoured_img = original_img.copy()
        if contours:
            cv2.drawContours(contoured_img, contours, -1, (0, 0, 255), contour_thickness)
        contoured_image = Image.fromarray(contoured_img)

        heatmap_overlay = Image.blend(Image.fromarray(original_img).convert("RGBA"),
                                      heatmap_image.convert("RGBA"), alpha=0.5).convert("RGB")
        heatmap_overlay_np = np.array(heatmap_overlay)
        if contours:
            cv2.drawContours(heatmap_overlay_np, contours, -1, (0, 0, 0), contour_thickness)
        overlay_img = Image.fromarray(heatmap_overlay_np)

        image_with_border = add_white_border(pil_image, 10)
        contours_with_border = add_white_border(overlay_img, 10)

        return pred_vec, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img
    except Exception as e:
        print("Error in import_and_predict_keras:", e)
        return None, None, None, None, None, None

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files.get('file')
        if not file or file.filename == "" or not allowed_file(file.filename):
            flash("Invalid or missing file")
            return redirect(request.url)

        filepath, filename = save_uploaded_file(file)
        pil_img = load_image_fix_orientation(filepath)
        pil_img = resize_if_large(pil_img)

        try:
            dry_weight_grams = float(request.form.get("dry_weight", 2800.0))
        except:
            dry_weight_grams = 2800.0

        image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(image_bgr, (224, 224)) / 255.0
        img_tensor = np.expand_dims(resized_image.astype(np.float32), axis=0)

        min_val, max_val = 2610, 3144
        dry_weight_norm = (dry_weight_grams - min_val) / (max_val - min_val)
        MIN_KN, MAX_KN = 2.13, 12.48

        if not all([class_model, strength_model, absorption_model]):
            flash("One or more models are missing.")
            return render_template("predict.html", filename=filename, error="Models missing")

        try:
            class_pred = run_tflite_inference(class_model, [img_tensor])
            class_label = int(np.argmax(class_pred[0]))

            if class_label == 3:
                return render_template("predict.html", filename=filename, not_brick=True)

            label_1 = 1 if class_label == 0 else 0
            label_2 = 1 if class_label == 1 else 0
            label_3 = 1 if class_label == 2 else 0
            tabular_input = np.array([[label_1, label_2, label_3, dry_weight_norm]], dtype=np.float32)

            strength_pred = run_tflite_inference(strength_model, [img_tensor, tabular_input])
            strength_denorm = float(strength_pred[0][0]) * (MAX_KN - MIN_KN) + MIN_KN

            tabular_input_abs = np.array([[dry_weight_grams, class_label + 1]], dtype=np.float32)
            absorption_pred = run_tflite_inference(absorption_model, [img_tensor, tabular_input_abs])
            absorption_real = float(absorption_pred[0][0])

            probs = (class_pred[0] * 100).tolist()
            return render_template("predict.html", filename=filename,
                                   strength=round(strength_denorm, 2),
                                   absorption=round(absorption_real, 2),
                                   probs=probs, class_idx=class_label)
        except Exception as e:
            flash(f"Prediction error: {e}")
            return render_template("predict.html", filename=filename, error=str(e))
    return render_template("predict.html")

@app.route("/cracks", methods=["GET", "POST"])
def cracks():
    if request.method == "POST":
        file = request.files.get('file')
        if not file or file.filename == "" or not allowed_file(file.filename):
            flash("Invalid or missing file")
            return redirect(request.url)

        filepath, filename = save_uploaded_file(file)
        pil_img = load_image_fix_orientation(filepath)
        pil_img = resize_if_large(pil_img)

        try:
            sensitivity = int(request.form.get("sensitivity", 9))
        except:
            sensitivity = 9

        if keras_model is None:
            flash("Crack detection model is missing.")
            return render_template("cracks.html", filename=filename, error="Model missing")

        try:
            preds, img_border, cont_border, heatmap_img, cont_img, overlay_img = import_and_predict_keras(
                pil_img, sensitivity=sensitivity)
            if preds is None:
                flash("Prediction failed.")
                return render_template("cracks.html", filename=filename, error="Prediction failed")

            predicted_class = int(np.argmax(preds))
            probs = (preds[0] * 100).tolist()

            def save_pil(pil_obj, suffix):
                uid = uuid.uuid4().hex[:8]
                fname = f"{uid}_{suffix}.png"
                path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
                pil_obj.save(path)
                return fname

            return render_template("cracks.html",
                                   filename=filename,
                                   probs=probs,
                                   predicted_class=predicted_class,
                                   fname_border=save_pil(img_border, "orig"),
                                   fname_contours=save_pil(cont_border, "contours"),
                                   fname_heatmap=save_pil(heatmap_img, "heatmap"),
                                   fname_contoured=save_pil(cont_img, "contoured"),
                                   fname_overlay=save_pil(overlay_img, "overlay"))
        except Exception as e:
            flash(f"Error processing image: {e}")
            return render_template("cracks.html", filename=filename, error=str(e))
    return render_template("cracks.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
