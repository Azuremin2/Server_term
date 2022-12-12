from flask import render_template, request, session, Blueprint, redirect, url_for
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fontprop = fm.FontProperties(fname='malgun.ttf')

main_blueprint = Blueprint('main', __name__, url_prefix='/main')


@main_blueprint.route('/', methods=['GET', 'POST'])
def index():
    return render_template('Mainpage.html')


@main_blueprint.route('/logout', methods=['GET', 'POST'])
def log_out():
    session.clear()
    return redirect(url_for('index'))


@main_blueprint.route('/menu', methods=['GET', 'POST'])
def menu():
    return render_template('Menu.html')


@main_blueprint.route('/view', methods=['GET', 'POST'])
def view():
    # request.script_root = url_for('index', _external=True)
    return render_template('View.html')



"""
@main.route('/opencv', methods=['GET', 'POST'])
def opencv():
    return render_template("Video.html")


video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier()
face_cascade.load(cv2.samples.findFile('static/haarcascade_frontalface_alt.xml'))


def gen(video):
    while True:
        success, image = video.read()
        frame_gray = cv2.cvtCaolor(image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        faces = face_cascade.detectMultiScale(frame_gray)

        for (x, y, w, h) in faces:
            center = (x + w // 2, y + h // 2)
            cv2.putText(image, "X: " + str(center[0]) + " Y: " + str(center[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 3)
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            faceROI = frame_gray[y:y + h, x:x + w]
        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@main.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video), mimetype='multipart/x-mixed-replace; boundary=frame')
"""
