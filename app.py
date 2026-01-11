from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import numpy as np
import os

# Create Flask App (IMPORTANT)
app = Flask(__name__, static_folder="static", static_url_path="/static")

# Upload configuration
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = load_model("Models/new_model.h5")

# Class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Prediction function
def predict_tumor(img_path, image_size=224):
    img = load_img(img_path, target_size=(image_size, image_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    if class_labels[class_index] == "notumor":
        result = "No Tumor Detected"
    else:
        result = f"Tumor Detected: {class_labels[class_index]}"

    return result, confidence

# Route
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Prediction
            result, confidence = predict_tumor(file_path)

            # Browser-accessible image URL
            image_url = url_for("static", filename=f"uploads/{filename}")

            # Debug (optional)
            print("Saved:", file_path)
            print("URL:", image_url)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_url=image_url
    )

# Run app
if __name__ == "__main__":
    app.run(debug=True)
