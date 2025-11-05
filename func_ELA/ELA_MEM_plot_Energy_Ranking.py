import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#-----------------------------------------------------------------------------------
def Plot_Energy_Ranking_TopN(path_read_info_ALLState:str, TopN:int, tick_params_size:int=20, path_save=False) -> None:
    #----------------------
    #read
    info_ALLState_df = pd.read_csv(path_read_info_ALLState, index_col=None, header=0, sep=',', encoding="utf-8", dtype={"state":str,"E":float,"SS":bool,"RA":str,"next_state":str})
    #---------------------
    #sort
    info_ALLState_df = info_ALLState_df.sort_values(by=['E'], ascending=False)
    #---------------------
    #fig, ax
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1,1,1)
    #---------------------
    #plot
    x = info_ALLState_df["state"][len(info_ALLState_df)-TopN:len(info_ALLState_df)]
    y = info_ALLState_df["E"][len(info_ALLState_df)-TopN:len(info_ALLState_df)]
    ax.barh(y=x, width=y, color="black") #https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.barh.html
    #---------------------
    #setting
    ax.tick_params(axis="x", labelsize = tick_params_size)#軸の大きさ
    ax.tick_params(axis="y", labelsize = tick_params_size)#軸の大きさ
    ax.set_xlabel("Energy",fontsize=30)
    ax.set_ylabel("States",fontsize=30)
    plt.margins(y=0.001) #Remove empty "padding" https://stackoverflow.com/questions/35225097/how-to-remove-empty-padding-in-matplotlib-barh-plot
    plt.tight_layout()
    #---------------------
    #save
    if path_save!=False:
        plt.savefig(path_save, bbox_inches='tight')
    #---------------------
    #show
    plt.show()



#-----------------------------------------------------------------------------------
def Plot_Energy_Ranking(path_read_info_ALLState:str, tick_params_size:int=7, path_save=False) -> None:
    #----------------------
    #read
    info_ALLState_df = pd.read_csv(path_read_info_ALLState, index_col=None, header=0, sep=',', encoding="utf-8", dtype={"state":str,"E":float,"SS":bool,"RA":str,"next_state":str})
    #---------------------
    #sort
    info_ALLState_df = info_ALLState_df.sort_values(by=['E'], ascending=False)
    #---------------------
    #fig, ax
    fig = plt.figure(figsize=(10, 25))
    ax = fig.add_subplot(1,1,1)
    #---------------------
    #plot
    x = info_ALLState_df["state"]
    y = info_ALLState_df["E"]
    ax.barh(y=x, width=y, color="black") #https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.barh.html
    #---------------------
    #setting
    ax.tick_params(axis="x", labelsize = 20)#軸の大きさ
    ax.tick_params(axis="y", labelsize = tick_params_size)#軸の大きさ
    ax.set_xlabel("Energy",fontsize=30)
    ax.set_ylabel("States",fontsize=30)
    plt.margins(y=0.001) #Remove empty "padding" https://stackoverflow.com/questions/35225097/how-to-remove-empty-padding-in-matplotlib-barh-plot
    plt.tight_layout()
    #---------------------
    #save
    if path_save!=False:
        plt.savefig(path_save, bbox_inches='tight')
    #---------------------
    #show
    plt.show()



#-----------------------------------------------------------------------------------
def Plot_Energy_Ranking_Heatmap_TopN(path_read_info_ALLState:str, TopN:int, tick_params_size:int=40, path_save=False) -> None:
    #----------------------
    #read
    info_ALLState_df = pd.read_csv(path_read_info_ALLState, index_col=None, header=0, sep=',', encoding="utf-8", dtype={"state":str,"E":float,"SS":bool,"RA":str,"next_state":str})

    #----------------------
    #sort
    info_ALLState_df = info_ALLState_df.sort_values(by=['E'], ascending=True)
    #state_TopN
    state_TopN_list = info_ALLState_df["state"][0:TopN].to_list()
    #heatmap_df
    heatmap_df = pd.DataFrame(data=np.array([list(state_str) for _, state_str in enumerate(state_TopN_list)]).T, columns=state_TopN_list, dtype=float)

    #----------------------
    #fig,ax
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1,1,1)
    
    #----------------------
    #plot heatmap
    sns.heatmap(data=heatmap_df, #ヒートマップを表現する2次元のデータ
                    vmin=None, #カラーマップの最小値
                    vmax=None, #カラーマップの最大値
                    cmap="binary", #Matplotlibのオブジェクトを指定し、カラーマップをデザイン
                    center=None, #colormapの中心値
                    robust=False,#True：カラーマップの外れ値の影響を考慮（vmin, vmax=Noneの場合）
                    annot=None, #True：セル上に値出力
                    fmt='.2g', #True：文字列フォーマットを指定
                    annot_kws=None, #テキストプロパティを指定し、セル上の出力値の書式デザイン (annot=Trueの場合）
                    linewidths=1, #枠線の太さ
                    linecolor='black', #枠線の色
                    cbar=False, #True：カラーバーを出力
                    cbar_kws=None, #カラーバーの書式デザイン
                    cbar_ax=None,#カラーバーの軸指定
                    square=True, #True：正方形のマップ出力
                    ax=ax, #マップの軸を指定
                    xticklabels=True, #True：pandas.DataFrameの列名をX軸として出力
                    yticklabels=False,#True：pandas.DataFrameの列名をY軸として出力
                    mask=None, #値をマスクするセルを指定
                    )
    #----------------------
    #setting
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, rotation=0)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.xticks(fontsize=tick_params_size,rotation=270)
    #----------------------
    #save
    if path_save!=False:
        plt.savefig(path_save, bbox_inches="tight")
    #----------------------
    #show
    plt.show()