import os

import base64
from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    render_template,
    redirect,
    url_for,
)
from visualizer import load_image, segment, to_base_64

app = Flask(__name__)

@app.route('/')
def index():
    segmented_image = generate_image()
    return render_template('index.html', **{ 
        'origin_image': to_base_64(load_image('images/facade.png')),
        'segmented_image': to_base_64(segmented_image),
    })  

@app.route('/upload_model', methods=['POST'])
def upload():
    file_obj = request.files['file']
    filename = file_obj.filename
    if filename[-2:] == 'h5' or filename[-3:] == 'tsv':
        file_obj.save('./model.h5')
    else:
        return '失敗しました', 400
    return redirect(url_for('index'))

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file_obj = request.files['file']
    filename = file_obj.filename
    if filename[-3:] in ('png', 'jpg'):
        file_obj.save('./images/facade.png')
    else:
        return '失敗しました', 400
    return redirect(url_for('index'))

@app.route('/reset', methods=['GET'])
def reset():
    segmented_image = generate_image()
    
    return render_template('index.html', **{ 
        'origin_image': to_base_64(load_image('images/facade.png')),
        'segmented_image': to_base_64(segmented_image),
    })  

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)

@app.route('/favicon.ico')
def send_image():
    return send_from_directory('images', 'favicon.png')

def generate_image():
    try:
        segmented_image = segment(load_image('images/facade.png'))
    except:
        import traceback
        print(traceback.format_exc())
        segmented_image = load_image('images/no_image.png')
    return segmented_image

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='9006')
