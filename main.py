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
    data_file_obj = request.files['datafile']
    index_file_obj = request.files['indexfile']
    meta_file_obj = request.files['metafile']
    data_filename = data_file_obj.filename
    index_filename = index_file_obj.filename
    meta_filename = meta_file_obj.filename
    if 'data' in data_filename:
        data_file_obj.save('./models/model.ckpt.data-00000-of-00001')
    else:
        return '失敗しました', 400
    if index_filename.endswith('index'):
        index_file_obj.save('./models/model.ckpt.index')
    else:
        return '失敗しました', 400
    if meta_filename.endswith('meta'):
        meta_file_obj.save('./models/model.ckpt.meta')
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
