from flask import Flask, render_template, request, flash, redirect, send_file
from werkzeug.utils import secure_filename
from main import getPrediction
import os
from io import BytesIO

UPLOAD_FOLDER = 'static/images/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__, static_folder="static")
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
@app.route('/', methods=['POST'])
def submit_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected for uploading')
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        processed_image_io = getPrediction(file_path)
        processed_image_io.seek(0)

        # Send the processed image to the client
        response = send_file(processed_image_io, mimetype='image/jpeg', as_attachment=False)

        # Delete the original file after sending the processed image
        os.remove(file_path)

        return response
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

if __name__ == "__main__":
    app.run()