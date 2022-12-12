from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import sklearn
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

fontprop = fm.FontProperties(fname='malgun.ttf')

app = Flask(__name__)


@app.route('/regress', methods=['GET'])
def regress_test():
    df = pd.read_csv("../csv/삼성전자주식데이터.csv")

    # 수집한 데이터의 위에서부터 5번째 줄까지
    head = print(df.head(5))
    # 데이터의 전체적인 통계
    info = print(df.info())
    # 컬럼별 null값 통계
    isnull = print(df.isnull().sum())



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



    # x값을 시가, 고가, 저자 중에서 선택할 수 있도록
    x_data = np.array(df['시가'], dtype=int)
    # y값은 종가로 고정
    y_data = np.array(df['종가'], dtype=int)
    x_data = (x_data.reshape(-1, 1))
    model = sklearn.linear_model.LinearRegression()
    model.fit(x_data, y_data)



    # 회귀 예측선
    pred = (model.coef_ * x_data + (model.intercept_))
    # 회귀식
    fomula = ("y = " + str(model.coef_) + "x" + " + " + str((model.intercept_)))
    print("단변량 회귀식 : ", fomula)
    n_score = model.score(x_data, y_data)
    print("단순선형회귀 평가점수 : ", n_score)


    # plt.scatter(x_data, y_data, c="blue")
    # plt.title("단변량 회귀선", fontproperties=fontprop)
    # plt.xlabel("시가", fontproperties=fontprop)
    # plt.ylabel("종가", fontproperties=fontprop)
    # plt.legend(loc='best', prop=fontprop)
    # plt.plot(x_data, pred, label="회귀선", c="red")
    # plt.title("", fontproperties=fontprop)
    # plt.xlabel("", fontproperties=fontprop)
    # plt.ylabel("", fontproperties=fontprop)
    # plt.legend(loc='best', prop=fontprop)


    # 시가, 고가, 저가 데이터를 하나의 x값으로
    m_x_data = np.array(df[['시가', '고가', '저가']], dtype=int)
    m_x_data = (m_x_data.reshape(-1, 3))
    # y값은 종가로 고정
    m_y_data = np.array(df['종가'], dtype=int)
    m_model = sklearn.linear_model.LinearRegression()
    m_model.fit(m_x_data, m_y_data)


    #다중선형회귀 회귀식, 평가점수
    m_pred = (m_model.coef_ * m_x_data + (m_model.intercept_))
    m_fomula = ("y = " + str(m_model.coef_) + "x1 x2 x3" + " " + str((m_model.intercept_)))
    print("다변량 회귀식 : ", m_fomula)
    m_score = m_model.score(m_x_data, m_y_data)
    print("다중선형회귀 평가점수 : ", m_score)


    start = int(request.args.get('_start'))
    mstart = int(request.args.get('_mstart'))
    mhigh = int(request.args.get('_mhigh'))
    mlow = int(request.args.get('_mlow'))

    predict_normal = model.predict([[start]])
    predict_multi = m_model.predict([[mstart, mhigh, mlow]])
    print("단변량 회귀 예측 종가 : ", predict_normal)
    print("다변량 회귀 예측 종가 : ", predict_multi)

    return render_template('regdata.html',
                           n_fomula=m_fomula,
                           n_score=n_score,
                           n_pred=predict_normal,
                           m_fomula=m_fomula,
                           m_score=m_score,
                           m_pred=predict_multi
                           )
