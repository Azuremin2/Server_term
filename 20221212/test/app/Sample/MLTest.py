from flask import Flask, render_template, request, Response, session
import pandas as pd
import numpy as np
import cv2
import sklearn
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

fontprop = fm.FontProperties(fname='malgun.ttf')

app = Flask(__name__)

app.secret_key = os.urandom(24)

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier()
face_cascade.load(cv2.samples.findFile("../static/haarcascade_frontalface_alt.xml"))


@app.route('/')
def index():
    if 'id' in session:
        return render_template('Mainpage.html')
    else:
        return render_template('Login.html')


@app.route('/menu', methods=['GET', 'POST'])
def menu():
    return render_template('Menu.html')


@app.route('/view', methods=['GET', 'POST'])
def view():
    return render_template('View.html')

@app.route('/regressinput', methods=['GET', 'POST'])
def regress_input():
    return render_template("reg1.html")


@app.route('/regress', methods=['GET', 'POST'])
def regress():
    df = pd.read_csv("../csv/삼성전자주식데이터.csv")

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

    plt.savefig("static/img/x_data.png")
    plt.show()












    # x값을 시가, 고가, 저자 중에서 선택할 수 있도록
    x_data = np.array(df['시가'], dtype=int)

    # y값은 종가로 고정
    y_data = np.array(df['종가'], dtype=int)
    x_data = (x_data.reshape(-1, 1))
    model = sklearn.linear_model.LinearRegression()
    model.fit(x_data, y_data)

    # 회귀식
    fomula = ("y = " + str(model.coef_) + "x" + " + " + str((model.intercept_)))
    print("단변량 회귀식 : ", fomula)
    n_score = model.score(x_data, y_data)
    print("단순선형회귀 평가점수 : ", n_score)

    plt.scatter(x_data, y_data, c="blue")
    plt.title("단변량 회귀선", fontproperties=fontprop)
    plt.xlabel("시가", fontproperties=fontprop)
    plt.ylabel("종가", fontproperties=fontprop)
    plt.legend(loc='best', prop=fontprop)

    # 회귀 예측선
    pred = (model.coef_ * x_data + (model.intercept_))

    plt.plot(x_data, pred, label="회귀선", c="red")
    plt.title("", fontproperties=fontprop)
    plt.xlabel("", fontproperties=fontprop)
    plt.ylabel("", fontproperties=fontprop)
    plt.legend(loc='best', prop=fontprop)

    plt.savefig("static/img/regress.png")
    plt.show()




#    # 시가, 고가, 저가 데이터를 하나의 x값으로
#    m_x_data = np.array(df[['시가', '고가', '저가']], dtype=int)
#    m_x_data = (m_x_data.reshape(-1, 3))

#    # y값은 종가로 고정
#    m_y_data = np.array(df['종가'], dtype=int)
#    m_model = sklearn.linear_model.LinearRegression()
#    m_model.fit(m_x_data, m_y_data)

#    다중선형회귀 회귀식, 평가점수
#    m_pred = (m_model.coef_ * m_x_data + (m_model.intercept_))

#    m_fomula = ("y = " + str(m_model.coef_) + "x1 x2 x3" + " " + str((m_model.intercept_)))
#    print("다변량 회귀식 : ", m_fomula)

#    m_score = m_model.score(m_x_data, m_y_data)
#    print("다중선형회귀 평가점수 : ", m_score)

    return render_template('regdata.html')



@app.route('/regress/result', methods=['GET', 'POST'])
def regress_result():
     df = pd.read_csv("../csv/삼성전자주식데이터.csv")

     # 수집한 데이터의 위에서부터 5번째 줄까지
     print(df.head(5))
     # 컬럼별 null값 통계
     print(df.isnull().sum())
     # 데이터의 전체적인 통계
     print(df.info())

     # 그래프 4개를 띄울건데, 크기가 x:10 y:8인
     fig = plt.figure()
     fig, ax = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

#     # 첫번째 위치
#     ax[0][0].scatter(df['시가'], df['종가'], c="blue")
#     ax[0][0].set_title("시가/종가", fontproperties=fontprop)
#     ax[0][0].set_xlabel("시가", fontproperties=fontprop)
#     ax[0][0].set_ylabel("종가", fontproperties=fontprop)
#     ax[0][0].legend(loc='best', prop=fontprop)

#     # 두번째 위치
#     ax[0][1].scatter(df['고가'], df['종가'], c="blue")
#     ax[0][1].set_title("고가/종가", fontproperties=fontprop)
#     ax[0][1].set_xlabel("고가", fontproperties=fontprop)
#     ax[0][1].set_ylabel("종가", fontproperties=fontprop)
#     ax[0][1].legend(loc='best', prop=fontprop)

#     # 세번째 위치
#     ax[1][0].scatter(df['저가'], df['종가'], c="blue")
#     ax[1][0].set_title("저가/종가", fontproperties=fontprop)
#     ax[1][0].set_xlabel("저가", fontproperties=fontprop)
#     ax[1][0].set_ylabel("종가", fontproperties=fontprop)
#     ax[1][0].legend(loc='best', prop=fontprop)

#     plt.savefig("static/img/x_data.png")
#     plt.show()


    # x값을 시가, 고가, 저자 중에서 선택할 수 있도록
     x_data = np.array(df['시가'], dtype=int)

     # y값은 종가로 고정
     y_data = np.array(df['종가'], dtype=int)
     x_data = (x_data.reshape(-1, 1))
     model = sklearn.linear_model.LinearRegression()
     model.fit(x_data, y_data)

     # 회귀식
     fomula = ("y = " + str(model.coef_) + "x" + " + " + str((model.intercept_)))
     print("단변량 회귀식 : ", fomula)
     n_score = model.score(x_data, y_data)
     print("단순선형회귀 평가점수 : ", n_score)


     plt.scatter(x_data, y_data, c="blue")
     plt.title("단변량 회귀선", fontproperties=fontprop)
     plt.xlabel("시가", fontproperties=fontprop)
     plt.ylabel("종가", fontproperties=fontprop)
     plt.legend(loc='best', prop=fontprop)

     # 회귀 예측선
     pred = (model.coef_ * x_data + (model.intercept_))

     plt.plot(x_data, pred, label="회귀선", c="red")
     plt.title("", fontproperties=fontprop)
     plt.xlabel("", fontproperties=fontprop)
     plt.ylabel("", fontproperties=fontprop)
     plt.legend(loc='best', prop=fontprop)
     plt.savefig("static/img/normal.png")
     plt.show()

     start = int(request.args.get('_start'))
     predict_normal = model.predict([[start]])
     print("단변량 회귀 예측 종가 : ", predict_normal)

     # 시가, 고가, 저가 데이터를 하나의 x값으로
     m_x_data = np.array(df[['시가', '고가', '저가']], dtype=int)
     m_x_data = (m_x_data.reshape(-1, 3))
     # y값은 종가로 고정
     m_y_data = np.array(df['종가'], dtype=int)
     m_model = sklearn.linear_model.LinearRegression()
     m_model.fit(m_x_data, m_y_data)

     # 다중선형회귀 회귀식, 평가점수
     m_pred = (m_model.coef_ * m_x_data + (m_model.intercept_))
     m_fomula = ("y = " + str(m_model.coef_) + "x1 x2 x3" + " " + str((m_model.intercept_)))
     print("다변량 회귀식 : ", m_fomula)
     m_score = m_model.score(m_x_data, m_y_data)
     print("다중선형회귀 평가점수 : ", m_score)


     mstart = int(request.args.get('_mstart'))
     mhigh = int(request.args.get('_mhigh'))
     mlow = int(request.args.get('_mlow'))


     predict_multi = m_model.predict([[mstart, mhigh, mlow]])
     print("다변량 회귀 예측 종가 : ", predict_multi)

     return render_template('regdata.html')



@app.route('/classifyinput', methods=['GET', 'POST'])
def classify_input():
    return render_template('Classify.html')


@app.route("/classify", methods=['GET', 'POST'])
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

    # for i in range(m_deep):
    #     model = sklearn.tree.DecisionTreeClassifier(max_depth=i + 1, random_state=0)
    #     model.fit(x_train, y_train)
    #     train_score[i] = sklearn.metrics.accuracy_score(y_train, model.predict(x_train))
    #     test_score[i] = sklearn.metrics.accuracy_score(y_test, model.predict(x_test))
    #
    # print(train_score)
    # print(test_score)

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
    plt.show()

    dept = int(request.form['max_depth'])

    # 클라이언트한테 트리의 깊이 값을 받음
    model = sklearn.tree.DecisionTreeClassifier(max_depth=dept, random_state=0)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_train)
    score = sklearn.metrics.accuracy_score(y_train, y_pred)
    print("분류 정확도 : ", score)

    # 클라이언트한테 입력받을 예측값들
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    height = int(request.form['height'])
    weight = int(request.form['weight'])
    fat = int(request.form['fat'])
    fastfood = int(request.form['fastfood'])
    vege = int(request.form['vege'])
    meal = int(request.form['meal'])
    snack = int(request.form['snack'])
    smoke = int(request.form['smoke'])
    water = int(request.form['water'])
    calory = int(request.form['calory'])
    activity = int(request.form['activity'])
    smartphone = int(request.form['smartphone'])
    drink = int(request.form['drink'])
    transport = int(request.form['transport'])

    pred = [[gender, age, height, weight, fat, fastfood, vege, meal, snack, smoke, water, calory, activity, smartphone,
             drink, transport]]
    pred_result = model.predict(pred)
    print("예측값 : ", pred)

    return render_template('Classifypred.html',
                           m_deep=m_deep,
                           dept=dept,
                           score=score,
                           data=pred_result)


@app.route('/clusterinput', methods=['GET', 'POST'])
def cluster_input():
    df = pd.read_csv("../csv/전국기상데이터.csv")

    # 불러온 데이터 확인
    print(df.head(5))

    # 결측치 발견
    print(df.isnull().sum())
    df = df.fillna(df.mean())
    # 결측값을 평균값으로 대체(개수 : 94 -> 95)
    df.info()

    # 정규분포로 변환 전 그래프(주석 입력 후 출력)
    plt.scatter(df['평균기온'], df['합계 강수량'], label="평균기온/합계강수량", c="blue")
    plt.title("평균기온과 합계 강수량", fontproperties=fontprop)
    plt.xlabel("평균기온", fontproperties=fontprop)
    plt.ylabel("합계 강수량", fontproperties=fontprop)
    plt.legend(loc='best', prop=fontprop)
    plt.savefig('static/img/before_distribute.png')
    plt.show()
    return render_template('Clusterdata.html')



@app.route('/cluster', methods=['GET', 'POST'])
def cluster():
    df = pd.read_csv("../csv/전국기상데이터.csv")

    # 불러온 데이터 확인
    print(df.head(5))
    # 결측치 발견
    print(df.isnull().sum())
    df = df.fillna(df.mean())
    # 결측값을 평균값으로 대체(개수 : 94 -> 95)
    df.info()

    # 정규분포로 변환 전 그래프(주석 입력 후 출력)
    plt.scatter(df['평균기온'], df['합계 강수량'], label="평균기온/합계강수량", c="blue")
    plt.title("평균기온과 합계 강수량", fontproperties=fontprop)
    plt.xlabel("평균기온", fontproperties=fontprop)
    plt.ylabel("합계 강수량", fontproperties=fontprop)
    plt.legend(loc='best', prop=fontprop)
    plt.savefig('static/img/before_distribute.png')
    plt.show()

    scaler = sklearn.preprocessing.StandardScaler()
    scaled = scaler.fit_transform(df[['평균기온', '합계 강수량']])
    cluster_df = pd.DataFrame(data=scaled, columns=['평균기온', '합계 강수량'])

    # 정규분포 변환한 그래프 비교 확인하도록

    # 정규분포 그래프를 먼저 찍어 준 이후 아래 코드 실행
    fig = plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    title = request.form['title']
    x = request.form['x']
    y = request.form['y']
    label = request.form['label']
    c = request.form['color']

    ax[0].scatter(df[x], df[y], label=label, c=c)
    ax[0].set_title(title, fontproperties=fontprop)
    ax[0].set_xlabel(x, fontproperties=fontprop)
    ax[0].set_ylabel(y, fontproperties=fontprop)
    ax[0].legend(loc='best', prop=fontprop)

    ax[1].scatter(cluster_df[x], cluster_df[y], label=label, c=c)
    ax[1].set_title(title, fontproperties=fontprop)
    ax[1].set_xlabel(x, fontproperties=fontprop)
    ax[1].set_ylabel(y, fontproperties=fontprop)
    ax[1].legend(loc='best', prop=fontprop)
    plt.savefig('static/img/distribute.png')
    plt.show()
    return render_template('Clusterdata.html')


@app.route('/clusterinput2', methods=['GET', 'POST'])
def cluster_input2():
    return render_template('Clusterpred.html')


@app.route('/cluster/result', methods=['GET', 'POST'])
def cluster_result():
    df = pd.read_csv("../csv/전국기상데이터.csv")

    # 불러온 데이터 확인
    head = df.head(5)
    # 결측치 발견
    isnull = df.isnull().sum()
    df = df.fillna(df.mean())
    # 결측값을 평균값으로 대체(개수 : 94 -> 95)
    info = df.info()

    # # 정규분포로 변환 전 그래프(주석 입력 후 출력)
    # plt.scatter(df['평균기온'], df['합계 강수량'], label="평균기온/합계강수량", c="blue")
    # plt.title("평균기온과 합계강수량", fontproperties=fontprop)
    # plt.xlabel("평균기온", fontproperties=fontprop)
    # plt.ylabel("합계 강수량", fontproperties=fontprop)
    # plt.legend(loc='best', prop=fontprop)
    # plt.savefig('static/img/before_distribute.png')
    # plt.show()

    scaler = sklearn.preprocessing.StandardScaler()
    scaled = scaler.fit_transform(df[['평균기온', '합계 강수량']])
    cluster_df = pd.DataFrame(data=scaled, columns=['평균기온', '합계 강수량'])

    # 정규분포 변환한 그래프 비교 확인하도록

    # 정규분포 그래프를 먼저 찍어 준 이후 아래 코드 실행
    # fig = plt.figure()
    # fig, ax = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    # ax[0].scatter(df['평균기온'], df['합계 강수량'], label="평균기온/합계강수량", c="blue")
    # ax[0].set_title("", fontproperties=fontprop)
    # ax[0].set_xlabel("평균기온", fontproperties=fontprop)
    # ax[0].set_ylabel("합계 강수량", fontproperties=fontprop)
    # ax[0].legend(loc='best', prop=fontprop)
    # ax[1].scatter(cluster_df['평균기온'], cluster_df['합계 강수량'], label="평균기온/합계강수량", c="blue")
    # ax[1].set_title("", fontproperties=fontprop)
    # ax[1].set_xlabel("평균기온", fontproperties=fontprop)
    # ax[1].set_ylabel("합계 강수량", fontproperties=fontprop)
    # ax[1].legend(loc='best', prop=fontprop)
    # plt.savefig('static/img/distribute.png')
    # plt.show()

    scaler = sklearn.preprocessing.StandardScaler()
    scaled = scaler.fit_transform(df[['평균기온', '합계 강수량']])
    cluster_df = pd.DataFrame(data=scaled, columns=['평균기온', '합계 강수량'])

    group = int(request.args.get('group'))

    # 생성할 그룹의 갯수를 클라이언트에게 입력받는다.
    model = sklearn.cluster.KMeans(n_clusters=group)

    # 군집화 에서는 데이터를 학습시킬 때 y값이 필요없다.
    model.fit(cluster_df)
    centroid = model.cluster_centers_

    # 각 데이터에 대한 군집화된 결과물을 '군집' 컬럼을 새로 만들어 그 곳에 저장한다.
    cluster_df['군집'] = model.labels_

    title = request.form['title']
    xlabel = request.form['x']
    ylabel = request.form['y']
    label = request.form['label']
    c = request.form['color']

    # 설정한 그룹의 갯수만큼 색으로 구별됨
    plt.scatter(cluster_df[f'{{x}}'], cluster_df[f'{{y}}'], label=label, c=cluster_df['군집'])
    plt.title(title, fontproperties=fontprop)
    plt.xlabel(xlabel, fontproperties=fontprop)
    plt.ylabel(ylabel, fontproperties=fontprop)
    plt.legend(loc='best', prop=fontprop)

    # 중심점 표시
    plt.scatter(centroid[:, 0], centroid[:, 1], c=c)
    plt.title(title, fontproperties=fontprop)
    plt.xlabel(xlabel, fontproperties=fontprop)
    plt.ylabel(ylabel, fontproperties=fontprop)
    plt.legend(loc='best', prop=fontprop)
    plt.savefig('static/img/scaling.png')
    plt.show()

    df['군집'] = cluster_df['군집']

    for i in range(group):
        plt.scatter(df[df['군집'] == i]['평균기온'], df[df['군집'] == i]['합계 강수량'], label=i, c=None)
        plt.title("평균기온과 합계 강수량", fontproperties=fontprop)
        plt.xlabel("평균기온", fontproperties=fontprop)
        plt.ylabel("합계 강수량", fontproperties=fontprop)
        plt.legend(loc='best', prop=fontprop)
    plt.savefig('static/img/cluster_result.png')
    plt.show()

    # for i in range(group):
    # data[i] = df[df['군집'] == i]['지점명'].unique()
    # print(data)

    # clu_area = group
    # for i in range(0, clu_area):
    #     globals()['area_{}'.format(i)] = df[df['군집'] == i]['지점명'].unique()
    #
    # for i in range(0, clu_area):
    #     print(globals()['area_{}'.format(i)])

    return render_template('Clusterpred.html')

@app.route('/opencv', methods=['POST'])
def opencv():
    return render_template("Video.html")

def gen(video):
    while True:
        success, image = video.read()
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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


@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=2204, debug=True)
