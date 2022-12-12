from flask import render_template, request, session, Blueprint, redirect, url_for
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fontprop = fm.FontProperties(fname='malgun.ttf')

reg_blueprint = Blueprint('reg', __name__, url_prefix='/reg')


@reg_blueprint.route('/reginfo', methods=['GET', 'POST'])
def reginfo():
    df = pd.read_csv("csv/삼성전자주식데이터.csv")

    # 수집한 데이터의 위에서부터 5번째 줄까지
    print(df.head(5))
    # 컬럼별 null값 통계
    print(df.isnull().sum())
    # 데이터의 전체적인 통계
    print(df.info())

    # 그래프 4개를 띄울건데, 크기가 x:10 y:8인
    fig = plt.figure()
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    # 첫번째 위치
    ax[0][0].scatter(df['시가'], df['종가'], c="blue")
    ax[0][0].set_title("시가/종가", fontproperties=fontprop)
    ax[0][0].set_xlabel("시가", fontproperties=fontprop)
    ax[0][0].set_ylabel("종가", fontproperties=fontprop)
    ax[0][0].legend(loc='best', prop=fontprop)
    # 두번째 위치
    ax[0][1].scatter(df['고가'], df['종가'], c="blue")
    ax[0][1].set_title("고가/종가", fontproperties=fontprop)
    ax[0][1].set_xlabel("고가", fontproperties=fontprop)
    ax[0][1].set_ylabel("종가", fontproperties=fontprop)
    ax[0][1].legend(loc='best', prop=fontprop)
    # 세번째 위치
    ax[1][0].scatter(df['저가'], df['종가'], c="blue")
    ax[1][0].set_title("저가/종가", fontproperties=fontprop)
    ax[1][0].set_xlabel("저가", fontproperties=fontprop)
    ax[1][0].set_ylabel("종가", fontproperties=fontprop)
    ax[1][0].legend(loc='best', prop=fontprop)
    plt.savefig("static/img/reg_x.png")
    plt.show()
    return render_template("regT.html")


@reg_blueprint.route('/regdata', methods=['GET', 'POST'])
def regdata():
    df = pd.read_csv("csv/삼성전자주식데이터.csv")

    # 수집한 데이터의 위에서부터 5번째 줄까지
    print(df.head(5))
    # 컬럼별 null값 통계
    print(df.isnull().sum())
    # 데이터의 전체적인 통계
    print(df.info())

    # 그래프 4개를 띄울건데, 크기가 x:10 y:8인
    fig = plt.figure()
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    # 첫번째 위치
    ax[0][0].scatter(df['시가'], df['종가'], c="blue")
    ax[0][0].set_title("시가/종가", fontproperties=fontprop)
    ax[0][0].set_xlabel("시가", fontproperties=fontprop)
    ax[0][0].set_ylabel("종가", fontproperties=fontprop)
    ax[0][0].legend(loc='best', prop=fontprop)
    # 두번째 위치
    ax[0][1].scatter(df['고가'], df['종가'], c="blue")
    ax[0][1].set_title("고가/종가", fontproperties=fontprop)
    ax[0][1].set_xlabel("고가", fontproperties=fontprop)
    ax[0][1].set_ylabel("종가", fontproperties=fontprop)
    ax[0][1].legend(loc='best', prop=fontprop)
    # 세번째 위치
    ax[1][0].scatter(df['저가'], df['종가'], c="blue")
    ax[1][0].set_title("저가/종가", fontproperties=fontprop)
    ax[1][0].set_xlabel("저가", fontproperties=fontprop)
    ax[1][0].set_ylabel("종가", fontproperties=fontprop)
    ax[1][0].legend(loc='best', prop=fontprop)
    plt.savefig("static/img/reg_x.png")
    plt.show()

    # # 단순선형회귀
    #
    # # x값을 시가, 고가, 저자 중에서 선택할 수 있도록
    # x_data = np.array(df['시가'], dtype=int)
    # # y값은 종가로 고정
    # y_data = np.array(df['종가'], dtype=int)
    # x_data = (x_data.reshape(-1, 1))
    # model = sklearn.linear_model.LinearRegression()
    # model.fit(x_data, y_data)
    # # 회귀식
    # fomula = ("y = " + str(model.coef_) + "x" + " + " + str((model.intercept_)))
    # print("단변량 회귀식 : ", fomula)
    # n_score = model.score(x_data, y_data)
    # print("단순선형회귀 평가점수 : ", n_score)
    # plt.scatter(x_data, y_data, c="blue")
    # plt.title("단변량 회귀선", fontproperties=fontprop)
    # plt.xlabel("시가", fontproperties=fontprop)
    # plt.ylabel("종가", fontproperties=fontprop)
    # plt.legend(loc='best', prop=fontprop)
    # # 회귀 예측선
    # pred = (model.coef_ * x_data + (model.intercept_))
    # plt.plot(x_data, pred, label="회귀선", c="red")
    # plt.title("", fontproperties=fontprop)
    # plt.xlabel("", fontproperties=fontprop)
    # plt.ylabel("", fontproperties=fontprop)
    # plt.legend(loc='best', prop=fontprop)
    # plt.savefig("static/img/reg1.png")
    # plt.show()
    #
    #
    # # 다중선형회귀(2개)
    #
    # # 사용자에게 (시가, 고가, 저가) 값 중 2개의 값을 받음
    # m_x_data = np.array(df[['시가', '고가']])
    # m_y_data = np.array(df['종가'])
    # m_x_data = (m_x_data.reshape(-1, 2))
    # m_x_train, m_x_test, m_y_train, m_y_test = sklearn.model_selection.train_test_split(m_x_data, m_y_data,
    #                                                                                     test_size=0.2, shuffle=True,
    #                                                                                     random_state=0)
    # m_model = sklearn.linear_model.LinearRegression(fit_intercept=True, n_jobs=None)
    # m_model.fit(m_x_train, m_y_train)
    # print("다변량회귀 평가점수", m_model.score(m_x_test, m_y_test))
    # m_pred = m_model.predict(m_x_data)
    # # 나머지 빈칸 사용자 입력값
    # # xlabel : 사용자 입력값 1
    # # ylabel : 사용자 입력값 2
    # # zlabel : 종가 고정
    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.gca(projection="3d")
    # ax.scatter(m_x_data[:, 0], m_x_data[:, 1], m_y_data, s=5, c='blue')
    # ax.set_xlabel("시가", fontproperties=fontprop)
    # ax.set_ylabel("고가", fontproperties=fontprop)
    # ax.set_zlabel("종가", fontproperties=fontprop)
    # plt.title("", fontproperties=fontprop)
    # plt.show()
    # ax.plot(m_x_data[:, 0], m_x_data[:, 1], m_pred, c='r', linestyle='solid')
    # plt.legend(loc='best', fontsize=10, prop=fontprop)
    # plt.savefig("static/img/reg2.png")
    # plt.show()
    # # 사용자에게 입력받을 값 1
    # val1 = 80000
    # # 사용자에게 입력받을 값 2
    # val2 = 85000
    # m_pred_result = m_model.predict([[val1, val2]])
    # print("다변량회귀 예측 종가 : ", m_pred_result, "원")
    #
    #
    # # 다중선형회귀(3개)
    #
    # # x값을 시가, 고가, 저가 데이터 모두 넣어주기
    # m_x_data = np.array(df[['시가', '고가', '저가']], dtype=int)
    # m_x_data = (m_x_data.reshape(-1, 3))
    # # y값은 종가로 고정
    # m_y_data = np.array(df['종가'], dtype=int)
    # m_model = sklearn.linear_model.LinearRegression()
    # m_model.fit(m_x_data, m_y_data)
    #
    # # 다중선형회귀 회귀식, 평가점수
    # m_pred = (m_model.coef_ * m_x_data + (m_model.intercept_))
    # m_fomula = ("y = " + str(m_model.coef_) + "x1 x2 x3" + " " + str((m_model.intercept_)))
    # print("다변량 회귀식 : ", m_fomula)
    # m_score = m_model.score(m_x_data, m_y_data)
    # print("다중선형회귀 평가점수 : ", m_score)

    return render_template('reg1.html')


@reg_blueprint.route('/reg1', methods=['GET', 'POST'])
def reg1():
    return render_template('reg1.html')


# 단순 선형 회귀
@reg_blueprint.route('/reg1pred', methods=['GET', 'POST'])
def reg1pred():
    df = pd.read_csv("csv/삼성전자주식데이터.csv")

    # x값을 시가, 고가, 저자 중에 하나의 값을 받는다.

    x = request.form['xdata']
    x_data = np.array(df[f'{x}'], dtype=int)

    # y값은 종가로 고정
    y_data = np.array(df['종가'], dtype=int)
    x_data = (x_data.reshape(-1, 1))

    # 선형회귀
    model = sklearn.linear_model.LinearRegression()
    model.fit(x_data, y_data)

    # 단순선형회귀 회귀식
    fomula = ("y = " + str(model.coef_) + "x" + " + " + str((model.intercept_)))
    print("단변량 회귀식 : ", fomula)

    # 단순선형회귀 평가점수
    score = model.score(x_data, y_data)
    print("단순선형회귀 평가점수 : ", score)

    # 회귀 예측선
    plt.scatter(x_data, y_data, c="blue")
    plt.title("단변량 회귀선", fontproperties=fontprop)
    plt.xlabel("시가", fontproperties=fontprop)
    plt.ylabel("종가", fontproperties=fontprop)
    plt.legend(loc='best', prop=fontprop)

    pred = (model.coef_ * x_data + (model.intercept_))
    plt.plot(x_data, pred, label="회귀선", c="red")
    plt.title("", fontproperties=fontprop)
    plt.xlabel("", fontproperties=fontprop)
    plt.ylabel("", fontproperties=fontprop)
    plt.legend(loc='best', prop=fontprop)
    plt.savefig("static/img/reg1pred.png")
    plt.show()

    # 단순선형회귀 예측값
    val = int(request.form['input'])
    predict_normal = model.predict([[val]])
    print("단변량 회귀 예측 종가 : ", predict_normal)

    return render_template('reg1.html',
                           fomula=fomula,
                           score=score,
                           pred=predict_normal
                           )


@reg_blueprint.route('/reg2', methods=['GET', 'POST'])
def reg2():
    return render_template('reg2.html')


# 다중선형회귀(x값 2개)
@reg_blueprint.route('/reg2pred', methods=['GET', 'POST'])
def reg2pred():
    df = pd.read_csv("csv/삼성전자주식데이터.csv")

    x1 = request.form['xdata1']
    x2 = request.form['xdata2']

    # x값을 (시가, 고가, 저가) 값 중 2개의 값을 받는다.
    m_x_data = np.array(df[[f'{x1}', f'{x2}']])
    m_y_data = np.array(df['종가'])
    m_x_data = (m_x_data.reshape(-1, 2))
    m_x_train, m_x_test, m_y_train, m_y_test = sklearn.model_selection.train_test_split(m_x_data,
                                                                                        m_y_data,
                                                                                        test_size=0.2,
                                                                                        shuffle=True,
                                                                                        random_state=0
                                                                                        )
    # 선형회귀
    m_model = sklearn.linear_model.LinearRegression(fit_intercept=True, n_jobs=None)
    m_model.fit(m_x_train, m_y_train)

    # 다변량회귀 회귀식

    # 다변량회귀 평가점수
    score = m_model.score(m_x_test, m_y_test)
    print("다변량회귀 평가점수", score)

    #
    pred = m_model.predict(m_x_data)

    # 나머지 빈칸 사용자 입력값
    # xlabel : 사용자 입력값 1
    # ylabel : 사용자 입력값 2
    # zlabel : 종가 고정

    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection="3d")
    ax.scatter(m_x_data[:f'{x1}', 0], m_x_data[:f'{x2}', 1], m_y_data, s=5, c='blue')
    ax.set_xlabel("시가", fontproperties=fontprop)
    ax.set_ylabel("고가", fontproperties=fontprop)
    ax.set_zlabel("종가", fontproperties=fontprop)
    plt.title("", fontproperties=fontprop)

    ax.plot(m_x_data[:f'{x1}', 0], m_x_data[:f'{x2}', 1], pred, c='r', linestyle='solid')
    plt.legend(loc='best', fontsize=10, prop=fontprop)
    plt.savefig("static/img/reg2pred.png")
    plt.show()

    # 사용자에게 입력받을 값 1
    val1 = request.form['input1']
    # 사용자에게 입력받을 값 2
    val2 = request.form['input2']

    # 다변량회귀 예측값
    pred_result = m_model.predict([[val1, val2]])
    print("다변량회귀 예측 종가 : ", pred_result, "원")

    return render_template('reg2.html',
                           score=score,
                           pred=pred_result
                           )

@reg_blueprint.route('/reg3', methods=['GET', 'POST'])
def reg3():
    return render_template('reg3.html')

# 다중선형회귀(3개)
@reg_blueprint.route('/reg3pred', methods=['GET', 'POST'])
def reg3pred():
    df = pd.read_csv("csv/삼성전자주식데이터.csv")

    # x값을 시가, 고가, 저가 데이터 모두 넣어주기
    m_x_data = np.array(df[['시가', '고가', '저가']], dtype=int)
    m_x_data = (m_x_data.reshape(-1, 3))

    # y값은 종가로 고정
    m_y_data = np.array(df['종가'], dtype=int)
    m_model = sklearn.linear_model.LinearRegression()
    m_model.fit(m_x_data, m_y_data)

    # 다중선형회귀 회귀식
    m_pred = (m_model.coef_ * m_x_data + (m_model.intercept_))
    m_fomula = ("y = " + str(m_model.coef_) + "x1 x2 x3" + " " + str((m_model.intercept_)))
    print("다변량 회귀식 : ", m_fomula)

    # 다중선형회귀 평가점수
    m_score = m_model.score(m_x_data, m_y_data)
    print("다중선형회귀 평가점수 : ", m_score)

    # 다변량회귀 예측값
    val1 = int(request.form['input1'])
    val2 = int(request.form['input2'])
    val3 = int(request.form['input3'])

    m_pred_result = m_model.predict([[val1, val2, val3]])
    print("다변량회귀 예측 종가 : ", m_pred_result, "원")

    return render_template('reg3.html',
                           fomula=m_fomula,
                           score=m_score,
                           pred=m_pred_result
                           )



