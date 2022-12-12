from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn import *
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fontprop = fm.FontProperties(fname='malgun.ttf')

app = Flask(__name__)


@app.route("/", methods=['GET'])
def home():
    return render_template('Input.html')


@app.route("/classify", methods=['GET'])
def classify():

    df = pd.read_csv("../csv/생활정보에따른비만정도.csv")

    print(df.head(5))
    print(df.info())
    print(df.isnull().sum())
    print(df["비만단계"].unique())

    x_data = df.drop(["비만단계"], axis="columns")
    y_data = df["비만단계"]

    # 테스트 데이터 / 학습데이터 분리도를 클라이언트가 설정할 수 있도록 : test_size 부분 0 ~ 1 사이의 값
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_data, y_data, test_size=0.2,
                                                                                shuffle=True, random_state=0)
    model = sklearn.tree.DecisionTreeClassifier(max_depth=None, random_state=0)
    model.fit(x_train, y_train)

    m_deep = model.get_depth()
    # 트리의 최대 깊이 확인
    print("트리의 최대 깊이 : ", m_deep)

    train_score = np.zeros(m_deep)
    test_score = np.zeros(m_deep)

    for i in range(m_deep):
        model = sklearn.tree.DecisionTreeClassifier(max_depth=i + 1, random_state=0)
        model.fit(x_train, y_train)
        train_score[i] = sklearn.metrics.accuracy_score(y_train, model.predict(x_train))
        test_score[i] = sklearn.metrics.accuracy_score(y_test, model.predict(x_test))

    print(train_score)
    print(test_score)

    plt.plot(train_score, label="훈련 데이터", c="red")
    plt.title("깊이에 따른 점수", fontproperties=fontprop)
    plt.xlabel("깊이", fontproperties=fontprop)
    plt.ylabel("점수", fontproperties=fontprop)
    plt.legend(loc='best', prop=fontprop)

    plt.plot(test_score, label="테스트 데이터", c="blue")
    plt.title("깊이에 따른 점수", fontproperties=fontprop)
    plt.xlabel("깊이", fontproperties=fontprop)
    plt.ylabel("점수", fontproperties=fontprop)
    plt.legend(loc='best', prop=fontprop)

    plt.savefig('static/img/scorebydept.png')

    # dept = int(request.args.get('_dept'))

    # 클라이언트한테 트리의 깊이 값을 받음
    model = sklearn.tree.DecisionTreeClassifier(max_depth=6, random_state=0)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_train)
    score = sklearn.metrics.accuracy_score(y_train, y_pred)
    print("분류 정확도 : ", score)

    # 클라이언트한테 입력받을 예측값들
    age = int(request.args.get('_age'))
    height = int(request.args.get('_height'))
    weight = int(request.args.get('_weight'))

    pred = [[1, age, height, weight, 1, 0, 2, 3, 20, 0, 2, 0, 0, 1, 10, 100]]
    pred_result = model.predict(pred)
    print("예측값 : ", pred)

    return render_template('Classifypred.html',
                            score=score,
                            data=pred_result)