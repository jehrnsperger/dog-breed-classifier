from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
from utils import detect_human_or_dog

UPLOAD_FOLDER = 'data/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            image_detection_answer = detect_human_or_dog(file_path)

            return redirect(url_for('uploaded_file', filename=filename, result=image_detection_answer))
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    result = request.args.get('result', None)
    return render_template('display.html', filename=filename, result=result)

@app.route('/data/images/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
