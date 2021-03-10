from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route("/one")
def one():
    return render_template('one.html')


@app.route('/change')
def change():
    render_template('change.html')


@app.route('/fusion')
def fusion():
    render_template('fusion.html')

    pass


@app.route('/chooseStyle')
def chooseStyle():
    render_template('chooseStyle.html')


@app.route('/chooseContent')
def chooseContent():
    render_template('chooseContent.html')


@app.route('/index')
def index():
    render_template('index.html')
    pass


if __name__ == '__main__':
    app.debug = False
    app.run()
