from flask import render_template, Blueprint
from flask_socketio import SocketIO, send

chat_blueprint = Blueprint('chat', __name__, url_prefix='/chat')


def socketio_init(socketio):

    @socketio.on('message', namespace='/chat')
    def handle_message(message):
        print("Received message: " + message)
        if message != "User connected!":
            send(message, broadcast=True)


@chat_blueprint.route('/', methods=['GET', 'POST'])
def index():
    return render_template("chat.html")

# @chat.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         session['name'] = request.form['name']
#         session['room'] = request.form['room']
#         return redirect(url_for('main.chat'))
#     return render_template('index.html')
