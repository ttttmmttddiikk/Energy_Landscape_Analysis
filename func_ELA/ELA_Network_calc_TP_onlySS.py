import numpy as np
import pandas as pd

from typing import Tuple, Union

from . import ELA_Network_functions

#-----------------------------------------------------------------------------------
# ④calc TP for all combinations of SS
#-----------------------------------------------------------------------------------
#-------------------------
def return_edge_df(info_ALLState_df:pd.DataFrame) -> pd.DataFrame:
    #edge_df
    edge_df = pd.DataFrame(index=[], columns=["state1","state2"])
    #calc
    for _, state1_use in enumerate(info_ALLState_df["state"]):
        #state2_use_list
        state2_use_list = ELA_Network_functions.return_AS_binary(state_str=state1_use)
        #state1_use_list
        state1_use_list = [state1_use for _ in range(len(state2_use_list))]
        #df_concat
        df_concat = pd.DataFrame(data=list(zip(*[state1_use_list,state2_use_list])), columns=["state1","state2"], index=None)
        #concat
        edge_df = pd.concat([edge_df,df_concat], axis=0)
    return edge_df

#-------------------------
import networkx as nx
def calc_TP_onlySS(info_ALLState_df:pd.DataFrame, diag_value:Union[int, str], 
                   path_save_TP_onlySS:str=False, path_save_TP_onlySS_name:str=False,
                   print_progress:str=False) -> pd.DataFrame:
    #-----------------------------------------------------------------------------------
    #----------------------
    #read
    #info_ALLState_df = pd.read_csv(path_read_info_ALLState, index_col=None, header=0, sep=',', encoding="utf-8", dtype={"state":str,"E":float,"SS":bool,"RA":str,"next_state":str})
    #----------------------
    #edge_df
    edge_df = return_edge_df(info_ALLState_df=info_ALLState_df)
    #----------------------
    #E_df
    E_df = info_ALLState_df[["state","E"]]
    #----------------------
    #edge_dfにE列を追加
    def func_add_E(row):
        state2 = row["state2"]
        E = E_df.query('state==@state2')["E"].values[0]
        row["E"] = E
        return row
    edge_df = edge_df.apply(func_add_E, axis=1)
    #----------------------
    #df -> Graoh
    # 辺の追加 (頂点も必要に応じて追加されます)
    G = nx.from_pandas_edgelist(df=edge_df, source='state1', target='state2', edge_attr=True, create_using=nx.MultiDiGraph())
    #----------------------
    # plot
    #nx.draw(G=G, pos=nx.spring_layout(G, k=0.7), with_labels=True, font_size=30, node_size=800, node_color="orange", font_color="black")
    #-----------------------------------------------------------------------------------
    #----------------------
    #df_EのEを大きい順に並べたリスト
    E_list = sorted(E_df["E"].to_list(), reverse=True)
    #----------------------
    #SSのstateのlist
    state_list = info_ALLState_df.query('SS==True')["state"].to_list()
    #TP_onlySS_df, TP_onlySS_name_df
    TP_onlySS_df = pd.DataFrame(data=[[False for j in range(len(state_list))] for i in range(len(state_list))], columns=state_list, index=state_list)
    TP_onlySS_name_df = pd.DataFrame(data=[[False for j in range(len(state_list))] for i in range(len(state_list))], columns=state_list, index=state_list)
    #----------------------
    #calc
    i = 0
    for _, E_use in enumerate(E_list):
        #----------------------
        #E属性の値で削除
        #https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.edges.html #G.edges(data=True)
        nodes_to_remove = [(edge[0],edge[1]) for edge in G.edges(data=True) if edge[2]["E"] == E_use]
        G.remove_edges_from(nodes_to_remove)
        #----------------------
        # plot
        #nx.draw(G=G, pos=nx.spring_layout(G, k=0.7), with_labels=True, font_size=18, node_size=800, node_color="orange", font_color="black")
        #plt.show()
        #----------------------
        #df_TPに追加
        for _, state1_use in enumerate(state_list):
            for _, state2_use in enumerate(state_list):
                if TP_onlySS_df.loc[state1_use, state2_use] == False:
                    if nx.has_path(G=G, source=state1_use, target=state2_use) == False: #初めてパスがなくなった時に、df_TPにEを追加 (&計算速度のために二つのifに分けている)
                        #----------------------
                        #register (TP_df)
                        TP_onlySS_df.loc[state1_use, state2_use] = E_use
                        #----------------------
                        #register (TP_name_df)
                        state_TP = info_ALLState_df.query('E==@E_use')["state"].to_list()[0]
                        TP_onlySS_name_df.loc[state1_use, state2_use] = state_TP
                        #----------------------
                        #print
                        i = i+1
                        if print_progress!=False:
                            print(i)
    #----------------------
    #TP_onlySS_df : 対角をE or 0で埋める
    for _, state_use in enumerate(state_list):
        if diag_value=="E":
            TP_onlySS_df.loc[state_use, state_use] = E_df.query('state==@state_use')["E"].values[0]
        elif diag_value==0:
            TP_onlySS_df.loc[state_use, state_use] = 0
        else:
            raise ValueError("diag_value must be E or 0")
    #TP_onlySS_name_df : 対角をstateで埋める
    for _, state_use in enumerate(state_list):
        TP_onlySS_name_df.loc[state_use, state_use] = state_use
    #----------------------
    #save
    if path_save_TP_onlySS!=False:
        TP_onlySS_df.to_csv(path_or_buf=path_save_TP_onlySS, header=True, index=True, encoding="utf-8")
    if path_save_TP_onlySS_name!=False:
        TP_onlySS_name_df.to_csv(path_or_buf=path_save_TP_onlySS_name, header=True, index=True, encoding="utf-8")
    #----------------------
    #return
    return TP_onlySS_df, TP_onlySS_name_df