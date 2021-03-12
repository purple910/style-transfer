import base64
import os

import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route("/one")
def one():
    return render_template('one.html')


@app.route('/change')
def change():
    return render_template('change.html')


@app.route('/fusion')
def fusion():
    data = request.args
    print(data)
    imgs = data.to_dict().get('data').split(',')
    print(imgs)
    return render_template('fusion.html')


@app.route('/style-merge', methods=['post'])
def merge():
    data = request.json
    dataPath = data[22:]
    imagedata = base64.b64decode(dataPath)
    if os.path.exists('static/img/result/touxiang.jpg'):
        os.remove('static/img/result/touxiang.jpg')
    with open('static/img/result/touxiang.jpg', 'wb') as file:
        file.write(imagedata)
        file.close()

    return jsonify(result='static/img/result/test.jpg')


@app.route('/chooseStyle')
def chooseStyle():
    png = os.listdir('static/img/style/png')
    imgs = []
    for i in png:
        child = os.path.join('/%s/%s' % ('static/img/style/png', i))
        img = {'url': child, 'label': i.split('.')[0]}
        imgs.append(img)
    # print(imgs)
    return render_template('chooseStyle.html', imgs=imgs)


@app.route('/index')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.debug = True
    app.run()
