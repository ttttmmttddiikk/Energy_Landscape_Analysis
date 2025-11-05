import numpy as np
import pandas as pd

from typing import Tuple

from . import ELA_Network_functions

#-----------------------------------------------------------------------------------
# ②calc if SS for all states
#-----------------------------------------------------------------------------------
#-------------------------
# input:state, E, & info_df(col: state, E) -> output:bool_if_it_is_SS
def check_SS(state_str:str, E:int, info_ALLState_df:pd.DataFrame) -> bool:
    # AS
    AS_list = ELA_Network_functions.return_AS_binary(state_str)
    # check
    for _, AS in enumerate(AS_list):
        # E_AS
        E_AS = info_ALLState_df.query('state == @AS')["E"].values[0]
        # compare E vs E_AS
        if E_AS < E:
            return False
    return True

#-------------------------
# input:info_df(col: state, E) -> output:info_df(col: state, E, bool_if_it_is_SS)
def calc_SS(info_ALLState_df:pd.DataFrame, path_save_info_SS:str=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #----------------------
    # calc SS
    info_ALLState_df["SS"] = info_ALLState_df.apply(lambda row: check_SS(state_str=row["state"],E=row["E"],info_ALLState_df=info_ALLState_df), axis=1)
    #----------------------
    # info_SS_df
    info_SS_df = info_ALLState_df.query('SS == True')
    # save
    if path_save_info_SS!=False:
        info_SS_df.to_csv(path_save_info_SS, header=True, index=False, encoding="utf-8")
    #----------------------
    # return
    return info_ALLState_df, info_SS_df