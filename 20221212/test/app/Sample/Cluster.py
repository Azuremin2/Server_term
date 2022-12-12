from flask import *
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fontprop = fm.FontProperties(fname='malgun.ttf')

app = Flask(__name__)


@app.route('/cluster', methods=['GET'])
def cluster_test():
    df = pd.read_csv("../csv/전국기상데이터.csv")

    # 불러온 데이터 확인
    head = df.head(5)
    # 결측치 발견
    isnull = df.isnull().sum()
    df = df.fillna(df.mean())
    # 결측값을 평균값으로 대체(개수 : 94 -> 95)
    info = df.info()


    # # 정규분포로 변환 전 그래프
    # plt.scatter(df['평균기온'], df['합계 강수량'], label="평균기온/합계강수량", c="blue")
    # plt.title("", fontproperties=fontprop)
    # plt.xlabel("평균기온", fontproperties=fontprop)
    # plt.ylabel("합계 강수량", fontproperties=fontprop)
    # plt.legend(loc='best', prop=fontprop)
    # plt.show()
    # scaler = sklearn.preprocessing.StandardScaler()
    # scaled = scaler.fit_transform(df[['평균기온', '합계 강수량']])
    # cluster_df = pd.DataFrame(data=scaled, columns=['평균기온', '합계 강수량'])
    # # 정규분포 변환한 그래프 비교 확인하도록

    # 정규분포 그래프를 먼저 찍어 준 이후 아래 코드 실행
    # fig = plt.figure()
    # fig, ax = plt.subplots(1, 2, figsize=(8,4) , constrained_layout=True)
    # ax[0].scatter( df['평균기온'], df['합계 강수량'], label = "평균기온/합계강수량", c = "blue")
    # ax[0].set_title("", fontproperties=fontprop)
    # ax[0].set_xlabel("평균기온", fontproperties=fontprop)
    # ax[0].set_ylabel("합계 강수량", fontproperties=fontprop)
    # ax[0].legend(loc='best', prop=fontprop)
    # ax[1].scatter( cluster_df['평균기온'], cluster_df['합계 강수량'], label = "평균기온/합계강수량", c = "blue")
    # ax[1].set_title("", fontproperties=fontprop)
    # ax[1].set_xlabel("평균기온", fontproperties=fontprop)
    # ax[1].set_ylabel("합계 강수량", fontproperties=fontprop)
    # ax[1].legend(loc='best', prop=fontprop)
    # plt.show()


    scaler = sklearn.preprocessing.StandardScaler()
    scaled = scaler.fit_transform(df[['평균기온', '합계 강수량']])
    cluster_df = pd.DataFrame(data=scaled, columns=['평균기온', '합계 강수량'])

    # 생성할 그룹의 갯수를 클라이언트에게 입력받는다.
    model = sklearn.cluster.KMeans(n_clusters=3)

    # 군집화 에서는 데이터를 학습시킬 때 y값이 필요없다.
    model.fit(cluster_df)
    centroid = model.cluster_centers_

    # 각 데이터에 대한 군집화된 결과물을 '군집' 컬럼을 새로 만들어 그 곳에 저장한다.
    cluster_df['군집'] = model.labels_


    # # 설정한 그룹의 갯수만큼 색으로 구별됨
    # plt.scatter(cluster_df['평균기온'], cluster_df['합계 강수량'], label="평균기온/합계강수량", c=cluster_df['군집'])
    # plt.title("", fontproperties=fontprop)
    # plt.xlabel("평균기온", fontproperties=fontprop)
    # plt.ylabel("합계 강수량", fontproperties=fontprop)
    # plt.legend(loc='best', prop=fontprop)
    #
    # # 중심점 표시
    # plt.scatter(centroid[:, 0], centroid[:, 1], c="red")
    # plt.title("", fontproperties=fontprop)
    # plt.xlabel("평균기온", fontproperties=fontprop)
    # plt.ylabel("합계 강수량", fontproperties=fontprop)
    # plt.legend(loc='best', prop=fontprop)


    # scaled_df에 담겨있는 군집화된 결과를 데이터프레임 df에 넣어줌
    df['군집'] = cluster_df['군집']


    cluster_0 = df[df['군집'] == 0]['지점명'].unique()
    cluster_1 = df[df['군집'] == 1]['지점명'].unique()
    cluster_2 = df[df['군집'] == 2]['지점명'].unique()

    print("[0번 군집 지역]")
    print(cluster_0)
    print("[1번 군집 지역]")
    print(cluster_1)
    print("[2번 군집 지역]")
    print(cluster_2)

    render_template('cluster_pred.html',
                    cluster_0=cluster_0,
                    cluster_1=cluster_1,
                    cluster_2=cluster_2
                    )