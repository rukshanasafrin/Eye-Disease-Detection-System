import os
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

from preprocess import preprocess_image
from gradcam import get_gradcam_heatmap, overlay_heatmap

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Ensure folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load trained model
model = load_model("models/eye_disease_model.h5")
class_names = ['Glaucoma','Cataract' ] #, 'Glaucoma', 'Normal'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    # Preprocess image
    preprocessed_filename = "preprocessed.jpg"
    preprocessed_path = os.path.join(app.config['UPLOAD_FOLDER'], preprocessed_filename)
    preprocessed_img = preprocess_image(input_path, save_path=preprocessed_path)

    x = np.expand_dims(preprocessed_img.astype("float32") / 255.0, axis=0)
    preds = model.predict(x)
    pred_label = class_names[np.argmax(preds)]

    # Grad-CAM (update last layer name if not using VGG)
    heatmap = get_gradcam_heatmap(model, x, last_conv_layer="block5_conv4")
    gradcam_filename = "gradcam.jpg"
    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
    overlay_heatmap(heatmap, preprocessed_path, output_path=gradcam_path)

    # Render result
    return render_template('result.html',
                           result=pred_label,
                           preprocess=f"/static/{preprocessed_filename}",
                           heatmap=f"/static/{gradcam_filename}")

if __name__ == '__main__':
    app.run(debug=True)
