import numpy as np
import pandas as pd

from . import ELA_Network_functions

#-----------------------------------------------------------------------------------
# ③calc RA for all states
#-----------------------------------------------------------------------------------
#-------------------------
# input:state, E, bool_if_it_is_SS, & info_df(col: state, E, bool_if_it_is_SS) -> output:RA State (final RA state)
def check_RA(state_str:str, E:int, SS:bool, info_ALLState_df:pd.DataFrame) -> str:
    if SS==False:
        # AS
        AS_list = ELA_Network_functions.return_AS_binary(state_str)
        # E_AS_list
        E_AS_list = [info_ALLState_df.query('state == @AS')["E"].values[0] for _, AS in enumerate(AS_list)]
        # E_min
        E_min = min(E_AS_list)
        # AS_min
        AS_min = AS_list[E_AS_list.index(E_min)]
        # SS_min
        SS_min = info_ALLState_df.query('state == @AS_min')["SS"].values[0]
        # next
        return check_RA(state_str=AS_min, E=E_min, SS=SS_min, info_ALLState_df=info_ALLState_df)
    elif SS==True:
        #return
        return state_str

#-------------------------
# input:state, E, bool_if_it_is_SS, & info_df(col: state, E, bool_if_it_is_SS) -> output:RA State (only 1 step RA state)
def return_next_state(state_str:str, E:int, SS:bool, info_ALLState_df:pd.DataFrame) -> str:
    if SS==False:
        # AS
        AS_list = ELA_Network_functions.return_AS_binary(state_str)
        # E_AS_list
        E_AS_list = [info_ALLState_df.query('state == @AS')["E"].values[0] for _, AS in enumerate(AS_list)]
        # E_min
        E_min = min(E_AS_list)
        # AS_min
        AS_min = AS_list[E_AS_list.index(E_min)]
        # next_state
        return AS_min
    elif SS==True:
        #return
        return state_str

#-------------------------
# input:info_df(col: state, E, bool_if_it_is_SS) -> output:info_df(col: state, E, bool_if_it_is_SS, RA, next_state)
def calc_RA(info_ALLState_df:pd.DataFrame, path_save_info_ALLState:str=False) -> pd.DataFrame:
    #----------------------
    # calc RA
    info_ALLState_df["RA"] = info_ALLState_df.apply(lambda row: check_RA(state_str=row["state"],E=row["E"],SS=row["SS"],info_ALLState_df=info_ALLState_df), axis=1)
    #----------------------
    # calc naxt_state
    info_ALLState_df["next_state"] = info_ALLState_df.apply(lambda row: return_next_state(state_str=row["state"],E=row["E"],SS=row["SS"],info_ALLState_df=info_ALLState_df), axis=1)
    #----------------------
    # save
    if path_save_info_ALLState!=False:
        info_ALLState_df.to_csv(path_save_info_ALLState, header=True, index=False, encoding="utf-8")
    #----------------------
    # return
    return info_ALLState_df