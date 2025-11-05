import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial' # setting font family

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

from typing import Tuple

from . import ELA_Network_functions

#-----------------------------------------------------------------------------------
# func
def collect_leaves(leave_use:str, Z_use :np.ndarray, dn_use:dict) -> list:
    #len_leaves
    len_leaves = len(dn_use["leaves"])
    #make leave_list
    leave_list = [leave_use]
    while len(list(filter(lambda leaf: leaf not in dn_use["leaves"], leave_list))) != 0:
        for leaf_not_SS in filter(lambda leaf: leaf not in dn_use["leaves"], leave_list):
            leave_list.remove(leaf_not_SS)
            leave_list.append(int(Z_use[leaf_not_SS-len_leaves][0]))
            leave_list.append(int(Z_use[leaf_not_SS-len_leaves][1]))
    #return
    return leave_list


#-----------------------------------------------------------------------------------
#disconnectivity graph
#-----------------------------------------------------------------------------------
def plot_DG(path_read_TP_onlySS:str, path_read_info_SS:str, featurenames_for_heatmap:list, SS_color_dict:dict, 
            fineTune_y:float, padding_x:float, ylim_min:Tuple[int, float], ylim_max:Tuple[int, float],
            state_text_fontsize:Tuple[int, float],DG_linewidth:Tuple[int, float],
            path_save_DG:str=False, path_save_DG_heatmap:str=False, show_fig:bool=True) -> None:
    #-------------
    #read
    TP_onlySS_df = pd.read_csv(path_read_TP_onlySS, index_col=0, header=0, sep=',', encoding="utf-8")
    TP_onlySS_df.index = TP_onlySS_df.index.astype('str').str.zfill(len(TP_onlySS_df.columns[0]))
    info_SS_df = pd.read_csv(path_read_info_SS, index_col=None, header=0, sep=',', encoding="utf-8", dtype={"state":str,"E":float,"SS":bool,"RA":str,"next_state":str})
    #-------------
    #var
    y_tick_labelsize = 38
    #-------------
    #plot
    if len(info_SS_df)==1:
        #----------------------------------------------
        #disconnectivity graph
        #----------------------------------------------
        #-------------
        #fig,ax
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1,1,1)
        #-------------
        #plot dot
        plt.plot(1, info_SS_df["E"].values[0], marker='o', markersize=20, color="black")
        #-------------
        #stateをtextとして書く
        #pos
        pos_x_use = 1
        pos_y_use = info_SS_df["E"].values[0]
        #fine tune
        ft_x = 0
        ft_y = fineTune_y
        #plot text
        ax.text(pos_x_use-ft_x, pos_y_use-ft_y, info_SS_df["state"].values[0], fontsize=state_text_fontsize, color="black", verticalalignment="top", horizontalalignment="center", fontweight='bold')
        #-------------
        #軸目盛り,軸ラベル,枠 を消す
        ax.tick_params(labelbottom=False, labelleft=True, labelright=False, labeltop=False)
        ax.tick_params(bottom=False, left=True, right=False, top=False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        #-------------
        #setting
        ax.set_ylim([ylim_min, ylim_max])
        ax.tick_params(labelsize=y_tick_labelsize)
        plt.xticks(fontsize=24)
        plt.ylabel("Energy",fontsize=45)
        plt.tight_layout()
        #save
        if path_save_DG!=False:
            plt.savefig(path_save_DG, bbox_inches='tight')
        #show
        if show_fig==False:
            plt.clf()
            plt.close()
        else:
            plt.show()

        #----------------------------------------------
        #Heatmap for disconnectivity graph
        #----------------------------------------------
        #-------------
        #check len of featurenames_for_heatmap
        if len(featurenames_for_heatmap) != len(info_SS_df["state"].values[0]):
            raise ValueError("len of featurenames_for_heatmap must be equal to len of features!!!")
        #-------------
        #fig,ax
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1,1,1)
        #-------------
        #make df for heatmap
        state_plot = [info_SS_df["state"].values[0]]
        list_of_heatmap_values = [[int(s_use) for _, s_use in enumerate(list(state_use))] for _, state_use in enumerate(state_plot)]
        heatmap_df = pd.DataFrame(data=list(zip(*list_of_heatmap_values)), columns=state_plot, index=None) #index=featurenames_for_heatmap
        #print(heatmap_df)
        #-------------
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
        #setting
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, rotation=0)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        plt.xticks(fontsize=40, rotation=270)
        #save
        if path_save_DG_heatmap!=False:
            plt.savefig(path_save_DG_heatmap, bbox_inches='tight')
        #show
        if show_fig==False:
            plt.clf()
            plt.close()
        else:
            plt.show()

    else:
        #----------------------------------------------
        #square-form distance matrix -> condensed_distance_matrix
        condensed_distance_matrix = squareform(TP_onlySS_df.values)
        #負の値を無くす ValueError: Linkage 'Z' contains negative distances.対策
        min_condensed_distance_matrix = min(condensed_distance_matrix)
        condensed_distance_matrix = condensed_distance_matrix - min_condensed_distance_matrix

        #----------------------------------------------
        #分類
        Z = linkage(y=condensed_distance_matrix, method='single', metric='euclidean', optimal_ordering=False)

        #----------------------------------------------
        #HC
        dn = dendrogram(Z, labels=TP_onlySS_df.index, no_plot=show_fig)

        #----------------------------------------------
        #-------------
        #default of pos_x_list, pos_y_list, state_dict
        pos_x_list = dict(zip(dn["leaves"], [cnt for cnt in range(len(dn["leaves"]))]))
        pos_y_list = dict(zip(dn["leaves"], [info_SS_df.query('state==@state_use')["E"].values[0] for _, state_use in enumerate(dn["ivl"])]))
        state_dict = dict(zip(dn["leaves"], dn["ivl"]))

        #-------------
        #make pos_list
        for cnt, bond_list in enumerate(Z):
            #extract
            leave1 = int(bond_list[0])
            leave2 = int(bond_list[1])
            #leave_next
            leave_next = len(dn["leaves"]) + cnt
            #-------------
            #pos_x_listに追加
            pos_x_list[leave_next] = (pos_x_list[leave1]+pos_x_list[leave2])/2
            #-------------
            #pos_y_listに追加
            leave1_list = collect_leaves(leave_use=leave1, Z_use=Z, dn_use=dn)
            leave2_list = collect_leaves(leave_use=leave2, Z_use=Z, dn_use=dn)
            #
            leave1_state_list = [state_dict[leaf] for _, leaf in enumerate(leave1_list)]
            leave2_state_list = [state_dict[leaf] for _, leaf in enumerate(leave2_list)]
            #min
            E_list = []
            for _, state1_group in enumerate(leave1_state_list):
                for _, state2_group in enumerate(leave2_state_list):
                    E_list.append(TP_onlySS_df.loc[state1_group,state2_group])
            min_E = min(E_list)
            # add
            pos_y_list[leave_next] = min_E
        
        #----------------------------------------------
        #disconnectivity graph
        #----------------------------------------------
        #-------------
        #fig,ax
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1,1,1)

        #-------------
        #plot kakukaku
        for cnt, bond_list in enumerate(Z):
            #extract
            leave1 = int(bond_list[0])
            leave2 = int(bond_list[1])
            #leave_next
            leave_next = len(dn["leaves"]) + cnt
            #plot line
            ax.plot([pos_x_list[leave1],pos_x_list[leave1]], [pos_y_list[leave1], pos_y_list[leave_next]], color="black", linewidth=DG_linewidth) #plot([x1, x2], [y1, y2])
            ax.plot([pos_x_list[leave1],pos_x_list[leave2]], [pos_y_list[leave_next], pos_y_list[leave_next]], color="black", linewidth=DG_linewidth) #plot([x1, x2], [y1, y2])
            ax.plot([pos_x_list[leave2],pos_x_list[leave2]], [pos_y_list[leave2], pos_y_list[leave_next]], color="black", linewidth=DG_linewidth) #plot([x1, x2], [y1, y2])

        #-------------
        #stateをtextとして書く
        for leaves_use, state_use in state_dict.items():
            #pos
            pos_x_use = pos_x_list[leaves_use]
            pos_y_use = pos_y_list[leaves_use]
            #fine tune
            ft_x = 0
            ft_y = fineTune_y
            #plot text
            ax.text(pos_x_use-ft_x, pos_y_use-ft_y, state_use, fontsize=state_text_fontsize, color="black", verticalalignment="top", horizontalalignment="center", fontweight='bold') #color=SS_color_dict[state_use]

        #-------------
        #軸目盛り,軸ラベル,枠 を消す
        ax.tick_params(labelbottom=False, labelleft=True, labelright=False, labeltop=False)
        ax.tick_params(bottom=False, left=True, right=False, top=False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)

        #-------------
        #setting        
        ax.set_xlim(left=0-padding_x) #!!!!!!
        ax.set_ylim([ylim_min, ylim_max])
        ax.tick_params(labelsize=y_tick_labelsize)#軸ラベルの大きさ
        plt.xticks(fontsize=24)
        #plt.xlabel("State",fontsize=18)
        plt.ylabel("Energy",fontsize=45)
        plt.tight_layout()
        #plt.title(title, fontsize=60, pad=40)
        #-------------
        #save
        if path_save_DG!=False:
            plt.savefig(path_save_DG, bbox_inches="tight")
        #-------------
        #show
        if show_fig==False:
            plt.clf()
            plt.close()
        else:
            plt.show()

        #----------------------------------------------
        #Heatmap for disconnectivity graph
        #----------------------------------------------
        #-------------
        #check len of featurenames_for_heatmap
        if len(featurenames_for_heatmap) != len(dn["ivl"][0]):
            raise ValueError("len of featurenames_for_heatmap must be equal to len of features!!!")
        #-------------
        #fig,ax
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1,1,1)
        #-------------
        #make df for heatmap
        list_of_heatmap_values = [[int(s_use) for _, s_use in enumerate(list(state_use))] for _, state_use in enumerate(dn["ivl"])]
        heatmap_df = pd.DataFrame(data=list(zip(*list_of_heatmap_values)), columns=dn["ivl"], index=None) #index=featurenames_for_heatmap
        #-------------
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
        #-------------
        #setting
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, rotation=0)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        plt.xticks(fontsize=40,rotation=270)
        #-------------
        #save
        if path_save_DG_heatmap!=False:
            plt.savefig(path_save_DG_heatmap, bbox_inches="tight")
        #-------------
        #show
        if show_fig==False:
            plt.clf()
            plt.close()
        else:
            plt.show()


