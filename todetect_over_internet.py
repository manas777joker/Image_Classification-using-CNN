import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash
from werkzeug.utils import secure_filename
from predict import *
from flask import send_file
import time
script = ''

UPLOAD_FOLDER = './predict'
ALLOWED_EXTENSIONS = set(['jpg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def result():
  return render_template('Result.html')

def start():
  myMain()
  return result()

@app.route('/')
def index():
  return render_template('index.html')

def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/return-file/')
def return_file():
  try:
    return send_file('output.txt',attachment_filename='output.txt',as_attachment=False)
  except Exception as e:
    return str(e)


@app.route('/image/', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    if 'file' not in request.files:
      return render_template('Nofile.html')
    file = request.files['file']
    if file.filename == '':
      return render_template('Nofile.html')
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], '1.jpg'))
      return start()
       
    return render_template('Invalid.html')
    
if __name__ == '__main__':
  app.run(debug=True)
