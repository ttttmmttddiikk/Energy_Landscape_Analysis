import numpy as np
import pandas as pd

from . import ELA_Network_functions

#-----------------------------------------------------------------------------------
# ③calc RA for all states
#-----------------------------------------------------------------------------------
#-------------------------
# input:とあるstate,E,SSかどうかのbool,列(state,E,SSかどうかのbool)のdf -> RAのState
def check_RA(state_str:str, E:int, SS:bool, info_ALLState_df:pd.DataFrame) -> str:
    #print(SS)
    if SS==False:
        #AS
        AS_list = ELA_Network_functions.return_AS_binary(state_str)
        #E_AS_list
        E_AS_list = [info_ALLState_df.query('state == @AS')["E"].values[0] for _, AS in enumerate(AS_list)]
        #E_min
        E_min = min(E_AS_list)
        #AS_min
        AS_min = AS_list[E_AS_list.index(E_min)]
        #SS_min
        SS_min = info_ALLState_df.query('state == @AS_min')["SS"].values[0]
        #next
        return check_RA(state_str=AS_min, E=E_min, SS=SS_min, info_ALLState_df=info_ALLState_df)
    elif SS==True:
        #print(state_str)
        #return
        return state_str

#-------------------------
# input:とあるstate,E,SSかどうかのbool,列(state,E,SSかどうかのbool)のdf -> RAのState
def return_naxt_state(state_str:str, E:int, SS:bool, info_ALLState_df:pd.DataFrame) -> str:
    #print(SS)
    if SS==False:
        #AS
        AS_list = ELA_Network_functions.return_AS_binary(state_str)
        #E_AS_list
        E_AS_list = [info_ALLState_df.query('state == @AS')["E"].values[0] for _, AS in enumerate(AS_list)]
        #E_min
        E_min = min(E_AS_list)
        #AS_min
        AS_min = AS_list[E_AS_list.index(E_min)]
        #next_state
        return AS_min
    elif SS==True:
        #print(state_str)
        #return
        return state_str

#-------------------------
# input:列(state,E,SSかどうかのbool)のdf -> 列(state,E,SSかどうかのbool,RAのState)のdf
def calc_RA(info_ALLState_df:pd.DataFrame, path_save_info_ALLState:str=False) -> pd.DataFrame:
    #----------------------
    #read
    #info_ALLState_df = pd.read_csv(path_read_info_ALLState, index_col=None, header=0, sep=',', encoding="utf-8", dtype={"state":str,"E":float,"SS":bool,"RA":str,"next_state":str})
    #----------------------
    #calc RA
    info_ALLState_df["RA"] = info_ALLState_df.apply(lambda row: check_RA(state_str=row["state"],E=row["E"],SS=row["SS"],info_ALLState_df=info_ALLState_df), axis=1)
    #----------------------
    #calc naxt_state
    info_ALLState_df["next_state"] = info_ALLState_df.apply(lambda row: return_naxt_state(state_str=row["state"],E=row["E"],SS=row["SS"],info_ALLState_df=info_ALLState_df), axis=1)
    #----------------------
    #save
    if path_save_info_ALLState!=False:
        info_ALLState_df.to_csv(path_save_info_ALLState, header=True, index=False, encoding="utf-8")
    #----------------------
    #return
    return info_ALLState_df