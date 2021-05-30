import base64
import json
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


@app.route("/vgg19")
def vgg19():
    return render_template('vgg19.html')


@app.route('/vgg19-merge', methods=['post'])
def vgg19_merge():
    style = request.values.get('style')
    content = request.values.get('content')
    dataPath = content[22:]
    imagedata = base64.b64decode(dataPath)
    if os.path.exists('static/img/content/vgg19.jpg'):
        os.remove('static/img/content/vgg19.jpg')
    with open('static/img/content/vgg19.jpg', 'wb') as file:
        file.write(imagedata)
        file.close()
    vgg19_style_transfer('static/img/content/vgg19.jpg', 'core/models/vgg19_paintA_' + style,
                         'static/img/result/test.jpg')
    return jsonify(result='static/img/result/test.jpg')


@app.route("/vgg16")
def vgg16():
    return render_template('vgg16.html')


@app.route('/vgg16-merge', methods=['post'])
def vgg16_merge():
    style = request.values.get('style')
    content = request.values.get('content')
    dataPath = content[22:]
    imagedata = base64.b64decode(dataPath)
    if os.path.exists('static/img/content/vgg16.jpg'):
        os.remove('static/img/content/vgg16.jpg')
    with open('static/img/content/vgg16.jpg', 'wb') as file:
        file.write(imagedata)
        file.close()
    vgg16_style_transfer('static/img/content/vgg16.jpg',
                         'core/models/vgg16_' + style + '/fast_style_transfer.ckpt-done',
                         'static/img/result/test.jpg')
    return jsonify(result='static/img/result/test.jpg')


@app.route('/change')
def change():
    return render_template('change.html')


@app.route('/change-merge', methods=['post'])
def change_merge():
    style = request.values.get('style')
    content = request.values.get('content')
    dataPath = content[22:]
    imagedata = base64.b64decode(dataPath)
    if os.path.exists('static/img/content/change.jpg'):
        os.remove('static/img/content/change.jpg')
    with open('static/img/content/change.jpg', 'wb') as file:
        file.write(imagedata)
        file.close()
    vgg19_style_transfer('static/img/content/change.jpg', 'core/models/vgg19_paintA_' + style,
                         'static/img/result/vgg19.jpg')
    vgg16_style_transfer('static/img/content/change.jpg',
                         'core/models/vgg16_' + style + '/fast_style_transfer.ckpt-done',
                         'static/img/result/vgg16.jpg')
    vgg16_style_transfer('static/img/content/change.jpg', 'core/model/' + style + '.ckpt-done',
                         'static/img/result/vgg.jpg')
    return jsonify(result=['static/img/result/vgg16.jpg', 'static/img/result/vgg19.jpg', 'static/img/result/vgg.jpg'])


@app.route('/times')
def times():
    return render_template('times.html')


import time


@app.route('/times-merge', methods=['post'])
def times_merge():
    style = request.values.get('style')
    content = request.values.get('content')
    dataPath = content[22:]
    imagedata = base64.b64decode(dataPath)
    if os.path.exists('static/img/content/change.jpg'):
        os.remove('static/img/content/change.jpg')
    with open('static/img/content/change.jpg', 'wb') as file:
        file.write(imagedata)
        file.close()
    result = []
    time.sleep(20)
    for i in range(10):
        # vgg19_style_transfer_times('static/img/content/change.jpg', str(i))
        result.append('static/img/result/mosaic_' + str(i) + '.jpg')
    return jsonify(result=result)


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
def style_merge():
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
    fusion_style_transfer('static/img/content/fusion.jpg', 'static/img/result/', styles)
    return jsonify(result='static/img/result/result_2.jpg')


@app.route('/customize', methods=['GET', 'POST'])
def customize():
    """
    多风格的保存
    自定义风格权重
    """
    data = request.args
    imgs = data.to_dict().get('data').split(',')
    session['customize'] = imgs
    b = [int(i) + 1 for i in imgs]
    return render_template('customize.html', imgs=b)


@app.route('/customize-weights', methods=['post'])
def customize_weights():
    """自定义权重"""
    js = json.loads(request.get_data(as_text=True))
    data = js["content"]
    weights = js["weights"]
    # print(data)
    dataPath = data[22:]
    imagedata = base64.b64decode(dataPath)
    if os.path.exists('static/img/content/fusion.jpg'):
        os.remove('static/img/content/fusion.jpg')
    with open('static/img/content/fusion.jpg', 'wb') as file:
        file.write(imagedata)
        file.close()
    styles = session.get('customize')
    custom_style_transfer('static/img/content/fusion.jpg', 'static/img/result/result_customize.jpg', styles, weights)
    return jsonify(result='static/img/result/result_customize.jpg')


@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    """
    多风格的保存
    自定义风格权重
    """
    data = request.args
    imgs = data.to_dict().get('data').split(',')
    session['dataset'] = imgs
    b = [int(i) + 1 for i in imgs]
    return render_template('dataset.html', imgs=b)


@app.route('/dataset-contrast', methods=['post'])
def dataset_contrast():
    """自定义权重"""
    js = json.loads(request.get_data(as_text=True))
    data = js["content"]
    weights = js["weights"]
    # print(data)
    dataPath = data[22:]
    imagedata = base64.b64decode(dataPath)
    if os.path.exists('static/img/content/fusion.jpg'):
        os.remove('static/img/content/fusion.jpg')
    with open('static/img/content/fusion.jpg', 'wb') as file:
        file.write(imagedata)
        file.close()
    styles = session.get('dataset')
    custom_style_transfer('static/img/content/fusion.jpg', 'static/img/result/result_customize.jpg', styles, weights)
    custom_style_transfer_train2014('static/img/content/fusion.jpg', 'static/img/result/dataset.jpg', styles, weights)
    return jsonify(result1='static/img/result/result_customize.jpg', result2='static/img/result/dataset.jpg')


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
