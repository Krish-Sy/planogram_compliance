import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import cv2
import pandas as pd
from shelf_code.shelf_analyzer import ShelfAnalyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MODEL_FOLDER'] = 'models'

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'image' not in request.files or 'excel' not in request.files:
            return 'No file part', 400
        
        image_file = request.files['image']
        excel_file = request.files['excel']
        
        if image_file.filename == '' or excel_file.filename == '':
            return 'No selected file', 400
        
        if allowed_file(image_file.filename) and allowed_file(excel_file.filename):
            image_filename = secure_filename(image_file.filename)
            excel_filename = secure_filename(excel_file.filename)

            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_filename)
            image_file.save(image_path)
            excel_file.save(excel_path)
            
            yolo_model_path = os.path.join(app.config['MODEL_FOLDER'], 'best.pt')
            output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'marked_output.png')
            
            analyzer = ShelfAnalyzer(yolo_model_path)
            product_info = analyzer.analyze_shelf(image_path)
            analyzer.mark_image(image_path, product_info, excel_path, output_image_path)

            return redirect(url_for('output_image', filename='marked_output.png'))

    return render_template('upload.html')

@app.route('/output/<filename>')
def output_image(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
