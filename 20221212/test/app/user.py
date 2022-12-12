from flask import render_template, request, session, Blueprint, redirect, url_for
from module import dbModule
from werkzeug.security import generate_password_hash, check_password_hash

user_blueprint = Blueprint('user', __name__, url_prefix='/user')


@user_blueprint.route('/login', methods=['GET', 'POST'])
def login():
    db = dbModule.Database()
    db.__init__()

    id = request.form['_id']
    pw = request.form['_pw']

    sql = f"select * from member where(id = '{id}')"

    row = db.executeOne(sql)
    db.commit()
    db.close()

    if check_password_hash(row['pw'], pw):  # row가 있을 경우
        session['loggedin'] = True
        session['id'] = row['id']
        session['name'] = row['name']
        session['age'] = row['age']
        session['height'] = row['h']
        session['weight'] = row['w']
        return redirect(url_for('main.index'))
    else:
        return redirect(url_for('ndex'))


@user_blueprint.route('/signpage', methods=['GET', 'POST'])
def sign_page():
    return render_template('SignUp.html')


@user_blueprint.route('/signup', methods=['GET', 'POST'])
def sign_up():
    db = dbModule.Database()
    db.__init__()

    name = request.form['_name']
    id = request.form['_id']
    pw = request.form['_pw']
    pw_hash = generate_password_hash(pw, salt_length=16)

    age = int(request.form['_age'])
    height = int(request.form['_height'])
    weight = int(request.form['_weight'])

    sql = f"insert into member(name,id,pw,age,h,w) values('{name}','{id}','{pw_hash}',{age},{height},{weight})"
    db.executeAll(sql)
    db.commit()
    db.close()
    return render_template('Login.html')


@user_blueprint.route('/changetemp', methods=['GET', 'POST'])
def change_temp():
    return render_template('userinfo.html')


@user_blueprint.route('/change', methods=['GET', 'POST'])
def change():
    db = dbModule.Database()
    db.__init__()

    id = session['id']
    name = request.form['_name']
    pw = request.form['_pw']
    pw_hash = generate_password_hash(pw, salt_length=16)
    age = int(request.form['_age'])
    height = int(request.form['_height'])
    weight = int(request.form['_weight'])

    sql = f"Update member set name='{name}',pw='{pw_hash}',age={age},h={height},w={weight} WHERE id='{id}'"
    db.executeAll(sql)
    db.commit()
    db.close()
    return render_template('View.html')
