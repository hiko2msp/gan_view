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
from visualizer import gen_image_b64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', **{
        'initlal_noises': [50] * 10,
        'image': gen_image_b64([50] * 10)
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

@app.route('/generate', methods=['GET'])
def generate():
    noise_list = [int(request.args.get(str(i))) for i in range(10)]
    return render_template('index.html', **{
        'initlal_noises': noise_list,
        'image': gen_image_b64(noise_list),
    })

@app.route('/reset', methods=['GET'])
def reset():
    return render_template('index.html', **{
        'initlal_noises': [50] * 10,
        'image': gen_image_b64([50] * 10)
    })

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)

@app.route('/favicon.ico')
def send_image():
    return send_from_directory('images', 'favicon.png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='9006')
