import base64
import os

import requests
from flask import Flask, render_template, request, jsonify, g, session

from core.main import *

app = Flask(__name__)
# 为flask程序的上下文,简单说来就是flask程序需要运行的环境变量等等
ctx = app.app_context()
# 激活上下文的操作,类似的,如果我们想要回收上下文,用ctx.pop()
ctx.push()
app.config['SECRET_KEY'] = 'flask'


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route("/one")
def one():
    return render_template('one.html')


@app.route('/one-merge', methods=['post'])
def one_merge():
    style = request.values.get('style')
    content = request.values.get('content')
    dataPath = content[22:]
    imagedata = base64.b64decode(dataPath)
    if os.path.exists('static/img/content/one.jpg'):
        os.remove('static/img/content/one.jpg')
    with open('static/img/content/one.jpg', 'wb') as file:
        file.write(imagedata)
        file.close()
    vgg19_style_transfer('static/img/content/one.jpg', 'core/models/vgg19_paintA_' + style, 'static/img/result/test.jpg')
    return jsonify(result='static/img/result/test.jpg')


@app.route('/change')
def change():
    return render_template('change.html')


@app.route('/change-merge', methods=['post'])
def change_merge():
    style = request.values.get('style')
    print(style)
    content = request.values.get('content')
    dataPath = content[22:]
    imagedata = base64.b64decode(dataPath)
    if os.path.exists('static/img/content/change.jpg'):
        os.remove('static/img/content/change.jpg')
    with open('static/img/content/change.jpg', 'wb') as file:
        file.write(imagedata)
        file.close()
    vgg19_style_transfer('static/img/content/change.jpg', 'core/models/vgg19_paintA_' + style, 'static/img/result/vgg19.jpg')
    vgg16_style_transfer('static/img/content/change.jpg', 'core/models/vgg16_' + style + '/fast_style_transfer.ckpt-done',
                         'static/img/result/vgg16.jpg')
    vgg16_style_transfer('static/img/content/change.jpg', 'core/model/' + style + '.ckpt-done',
                         'static/img/result/vgg.jpg')
    return jsonify(result=['static/img/result/vgg16.jpg', 'static/img/result/vgg19.jpg', 'static/img/result/vgg.jpg'])


@app.route('/fusion', methods=['GET', 'POST'])
def fusion():
    """多风格的保存"""
    data = request.args
    # print(data)
    imgs = data.to_dict().get('data').split(',')
    # print(imgs)
    session['imgs'] = imgs
    # global index_add_counter
    # index_add_counter = imgs
    return render_template('fusion.html')


@app.route('/style-merge', methods=['post'])
def merge():
    """多风格转换返回值"""
    data = request.json
    dataPath = data[22:]
    imagedata = base64.b64decode(dataPath)
    if os.path.exists('static/img/content/fusion.jpg'):
        os.remove('static/img/content/fusion.jpg')
    with open('static/img/content/fusion.jpg', 'wb') as file:
        file.write(imagedata)
        file.close()
    styles = session.get('imgs')
    print(styles)
    # print(index_add_counter)
    fusion_style_transfer('static/img/content/fusion.jpg', 'static/img/result/', styles)
    return jsonify(result='static/img/result/result_2.jpg')


@app.route('/chooseStyle')
def chooseStyle():
    """多风格选择"""
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
