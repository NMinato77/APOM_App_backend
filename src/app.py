from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import os
import numpy as np
import cupy as cp
from werkzeug.utils import secure_filename
import tifffile
import reconstruction
from queue import Queue

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = r'E:\apom_app_temp\uploads'
RESULT_FOLDER = r'E:\apom_app_temp\results'
ALLOWED_EXTENSIONS = {'tif', 'tiff', 'fits'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# ✅ Queue を用いて進捗を管理
progress_queue = Queue()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    def progress_message(message):
        progress_queue.put(message)

    if 'image' not in request.files or 'profile' not in request.files or 'background' not in request.files:
        return jsonify({"error": "Missing file part"}), 400

    image = request.files['image']
    profile = request.files['profile']
    background = request.files['background']

    if image.filename == '' or profile.filename == '' or background.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if image and allowed_file(image.filename) and allowed_file(profile.filename) and allowed_file(background.filename):
        image_filename = secure_filename(image.filename)
        profile_filename = secure_filename(profile.filename)
        background_filename = secure_filename(background.filename)

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        profile_path = os.path.join(app.config['UPLOAD_FOLDER'], profile_filename)
        background_path = os.path.join(app.config['UPLOAD_FOLDER'], background_filename)

        image.save(image_path)
        profile.save(profile_path)
        background.save(background_path)

        progress_message("Files uploaded successfully...")

        fps = float(request.form.get('fps', 30))
        v = float(request.form.get('v', 1.0))
        wavelength = int(request.form.get('wavelength', 488))
        polarity = int(request.form.get('polarity', 0))
        calibration = request.form.get('calibration') == 'true'
        save_tif = request.form.get('save_tif') == 'true'
        save_folder = app.config['RESULT_FOLDER']

        progress_message("Starting reconstruction...")

        result_path = reconstruction.reconstruction(fps, v, wavelength, 
                                                    image_path, profile_path, background_path, 
                                                    polarity, calibration, save_tif,
                                                    save_folder)
        progress_message("Reconstruction completed...")

        return jsonify({
            "message": "File processed successfully",
            "download_url": f"/download/{os.path.basename(result_path)}"
        }), 200

    return jsonify({"error": "Invalid file format"}), 400

@app.route('/progress')
def progress():
    def generate():
        while True:
            message = progress_queue.get()  # ✅ Queue からメッセージを取得
            yield f"data: {message}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    return send_file(result_path, as_attachment=True)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)