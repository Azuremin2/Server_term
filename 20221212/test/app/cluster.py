from flask import render_template, request, session, Blueprint, redirect, url_for
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fontprop = fm.FontProperties(fname='malgun.ttf')

clust_blueprint = Blueprint('clust', __name__, url_prefix='/main')


@clust_blueprint.route('/clusterinfo', methods=['GET', 'POST'])
def clusterinfo():
    return render_template("clusT.html")


@clust_blueprint.route('/clusterdata', methods=['GET', 'POST'])
def clusterdata():
    df = pd.read_csv("csv/전국기상데이터.csv")

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

@clust_blueprint.route('/cluster', methods=['GET', 'POST'])
def cluster():
    df = pd.read_csv("csv/전국기상데이터.csv")

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
    return render_template("Clusterpred.html")

@clust_blueprint.route('/clusterpred', methods=['GET', 'POST'])
def clusterpred():
    df = pd.read_csv("csv/전국기상데이터.csv")

    # 불러온 데이터 확인
    head = df.head(5)
    # 결측치 발견
    isnull = df.isnull().sum()
    df = df.fillna(df.mean())
    # 결측값을 평균값으로 대체(개수 : 94 -> 95)
    info = df.info()

    scaler = sklearn.preprocessing.StandardScaler()
    scaled = scaler.fit_transform(df[['평균기온', '합계 강수량']])
    cluster_df = pd.DataFrame(data=scaled, columns=['평균기온', '합계 강수량'])

    group = int(request.form['group'])

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
