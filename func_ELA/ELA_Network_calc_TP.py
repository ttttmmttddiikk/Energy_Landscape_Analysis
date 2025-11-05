import numpy as np
import pandas as pd

from typing import Tuple, Union

from . import ELA_Network_functions

#-----------------------------------------------------------------------------------
# ④calc TP for all combinations
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
        #concat_df
        concat_df = pd.DataFrame(data=list(zip(*[state1_use_list,state2_use_list])), columns=["state1","state2"], index=None)
        #remove empty df
        list_df_concat = [df for df in [edge_df, concat_df] if not df.empty]
        #concat
        edge_df = pd.concat(list_df_concat, axis=0)
    return edge_df

#-------------------------
import networkx as nx
def calc_TP(info_ALLState_df:pd.DataFrame, diag_value:Union[int, str], 
            path_save_TP:str=False, path_save_TP_name:str=False,
            print_progress:str=False) -> pd.DataFrame:
    #-----------------------------------------------------------------------------------
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
    G = nx.from_pandas_edgelist(df=edge_df, source='state1', target='state2', edge_attr=True, create_using=nx.MultiDiGraph())
    #----------------------
    # plot
    #nx.draw(G=G, pos=nx.spring_layout(G, k=0.7), with_labels=True, font_size=30, node_size=800, node_color="orange", font_color="black")
    #-----------------------------------------------------------------------------------
    #----------------------
    #E_dfのEを大きい順に並べたリスト
    E_list = sorted(E_df["E"].to_list(), reverse=True)
    #----------------------
    #TP_df, TP_name_df
    columnNames_list = E_df["state"].to_list()
    TP_df = pd.DataFrame(data=[[None for j in range(len(columnNames_list))] for i in range(len(columnNames_list))], columns=columnNames_list, index=columnNames_list)
    TP_name_df = pd.DataFrame(data=[[None for j in range(len(columnNames_list))] for i in range(len(columnNames_list))], columns=columnNames_list, index=columnNames_list)
    #----------------------
    #calc
    i = 0
    for _, E_use in enumerate(E_list):
        #----------------------
        #E属性の値で削除
        nodes_to_remove = [(edge[0],edge[1]) for edge in G.edges(data=True) if edge[2]["E"] == E_use]
        G.remove_edges_from(nodes_to_remove)
        #----------------------
        # plot
        #nx.draw(G=G, pos=nx.spring_layout(G, k=0.7), with_labels=True, font_size=18, node_size=800, node_color="orange", font_color="black")
        #plt.show()
        #----------------------
        #df_TPに追加
        state_list = E_df["state"].to_list()
        for _, state1_use in enumerate(state_list):
            for _, state2_use in enumerate(state_list):
                if TP_df.loc[state1_use, state2_use] is None: #まだTP_dfにEが登録されていない場合のみ
                    if nx.has_path(G=G, source=state1_use, target=state2_use) == False: #初めてパスがなくなった時に、TP_dfにEを追加 (&計算速度のために二つのifに分けている)
                        #----------------------
                        #register (TP_df)
                        TP_df.loc[state1_use, state2_use] = E_use
                        #----------------------
                        #register (TP_name_df)
                        state_TP = info_ALLState_df.query('E==@E_use')["state"].to_list()[0]
                        TP_name_df.loc[state1_use, state2_use] = state_TP
                        #----------------------
                        #print
                        i = i+1
                        if print_progress!=False:
                            print(i)
    #----------------------
    #TP_df : 対角をE or 0で埋める
    for _, state_use in enumerate(E_df["state"].to_list()):
        if diag_value=="E":
            TP_df.loc[state_use, state_use] = E_df.query('state==@state_use')["E"].values[0] #対角をEで埋める
        elif diag_value==0:
            TP_df.loc[state_use, state_use] = 0 #対角を0で埋める
        else:
            raise ValueError("diag_value must be E or 0")
    #TP_name_df : 対角をstateで埋める
    for _, state_use in enumerate(state_list):
        TP_name_df.loc[state_use, state_use] = state_use
    #----------------------
    #save
    if path_save_TP!=False:
        TP_df.to_csv(path_save_TP, header=True, index=True, encoding="utf-8")
    if path_save_TP_name!=False:
        TP_name_df.to_csv(path_save_TP_name, header=True, index=True, encoding="utf-8")
    #----------------------
    #return
    return TP_df, TP_name_df