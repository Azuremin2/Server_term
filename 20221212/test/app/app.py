from flask import Flask, render_template, Blueprint, request, session, jsonify
from flask_socketio import SocketIO, send
from module import dbModule
from user import user_blueprint
from main import main_blueprint
from reg import reg_blueprint
from classify import clsf_blueprint
from cluster import clust_blueprint


from chat import get_response

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to quess...'

socketio = SocketIO(app, cors_allowed_origins="*")

app.register_blueprint(user_blueprint)
app.register_blueprint(main_blueprint)
app.register_blueprint(reg_blueprint)
app.register_blueprint(clsf_blueprint)
app.register_blueprint(clust_blueprint)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("Login.html")


@socketio.on('message')
def handle_message(message):
    print("Received message: " + message)
    if message != "User connected!":
        send(message, broadcast=True)


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    name = session['name']
    return render_template("chat.html", name=name)


@app.route('/chat_save', methods=['GET', 'POST'])
def save_chat():
    db = dbModule.Database()
    db.__init__()

    username = request.json['username']
    message = request.json['message']

    sql = f"insert into chat_data(timelog, username, message) values(Now(), '{username}', '{message}')"
    db.executeAll(sql)
    db.commit()
    db.close()
    return '', 204


@app.route('/chatlog', methods=['GET', 'POST'])
def chat_log():
    db = dbModule.Database()
    db.__init__()

    sql = "select timelog, username, message from chat_data"

    rows = db.executeAll(sql)
    db.commit()
    db.close()

    return render_template('chatlog.html', list=rows)


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)


if __name__ == "__main__":
    socketio.run(app, host="localhost", port=8080, debug=True)
    # app.run(host="localhost", port=5000, debug=True)
