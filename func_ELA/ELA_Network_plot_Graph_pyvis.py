import numpy as np
import pandas as pd

from . import ELA_Network_functions

#-----------------------------------------------------------------------------------
from pyvis.network import Network

def plot_Graph_by_pyvis(path_read_info_ALLState:str, SS_color_dict:dict, path_save:str=False) -> None:
    #----------------------
    # read
    info_ALLState_df = pd.read_csv(path_read_info_ALLState, index_col=None, header=0, sep=',', encoding="utf-8", dtype={"state":str,"E":float,"SS":bool,"RA":str,"next_state":str})

    #----------------------
    # instance
    net = Network(height='800px', width='1500px')

    #----------------------
    # E_max, E_min
    E_max = max(info_ALLState_df["E"].to_list())
    E_min = min(info_ALLState_df["E"].to_list())

    #----------------------
    # add node
    for _, row_series in info_ALLState_df.iterrows():
        # extract
        state_use = row_series["state"]
        E_use = row_series["E"]
        # RA color determined by gradation based on the magnitude of E
        color_use = SS_color_dict[row_series["RA"]]
        color_use_gradetion = ELA_Network_functions.get_color_of_gradation_from_int_to_white(base_color=color_use, use_num=E_use, min=E_min, max=E_max)
        # add node
        if row_series["SS"] == True:
            net.add_node(state_use, color=color_use_gradetion, shape='square', size=20) 
        else:
            net.add_node(state_use, color=color_use_gradetion, shape='dot', size=20)

    #----------------------
    # add edge
    for _, row_series in info_ALLState_df.iterrows():
        # extract
        state_use = row_series["state"]
        next_state_use = row_series["next_state"]
        RA_use = row_series["RA"]
        AS_list = ELA_Network_functions.return_AS_binary(state_str=state_use)
        # RA color determined by gradation based on the magnitude of E
        color_use = SS_color_dict[RA_use]
        # add edge
        for _, AS_use in enumerate(AS_list):
            # AS_next_state_use
            AS_next_state_use = info_ALLState_df.query('state==@AS_use')["next_state"].values[0]
            # add
            if AS_next_state_use == state_use:
                pass
            elif AS_use == next_state_use:
                net.add_edge(state_use, AS_use, color=color_use, width=5)
            else:
                pass
    
    #----------------------
    # change the node label font size
    for n in net.nodes:
        n["size"] = 20
        n["font"]={"size": 40, "bold":True}
    #----------------------
    # write
    if path_save!=False:
        net.show(path_save, notebook=False)

