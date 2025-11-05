import numpy as np
import pandas as pd

from . import ELA_Network_functions

#-----------------------------------------------------------------------------------
# ①calc E of all state from h and J
#-----------------------------------------------------------------------------------
#-------------------------
# input:path of h & J -> output:df(col:state,E)
def calc_E_ALLstate(path_read_h:str, path_read_J:str) -> pd.DataFrame:
    #--------------
    # read csv
    h_1darray = pd.read_csv(path_read_h, index_col=None, header=0, sep=',', encoding="utf-8").values.flatten()
    J_2darray = pd.read_csv(path_read_J, index_col=None, header=0, sep=',', encoding="utf-8").values
    #--------------
    # check
    ELA_Network_functions.check_shape(h_1darray,J_2darray)
    #--------------
    # input:h & J -> output:df of all state
    info_ALLState_df = ELA_Network_functions.make_ALLstate(h_1darray,J_2darray)
    #--------------
    # input:state,h,J -> output:E & Apply the function to all rows of the DataFrame’s “state” column and add the result as a new “E” column
    info_ALLState_df["E"] = info_ALLState_df.apply(lambda row: ELA_Network_functions.calc_E_1state(state_str=row["state"], h_1darray=h_1darray, J_2darray=J_2darray), axis=1)
    #--------------
    # return
    return info_ALLState_df

#-------------------------
# input:value of h & J -> output:df(col:state,E)
def calc_E_ALLstate_from_h_and_J(h_1darray:np.array, J_2darray:np.array) -> pd.DataFrame:
    # check
    ELA_Network_functions.check_shape(h_1darray,J_2darray)
    # input:h & J -> output:df of all state
    info_ALLState_df = ELA_Network_functions.make_ALLstate(h_1darray,J_2darray)
    # input: state,h,J -> Apply the function to all rows of the DataFrame’s “state” column and add the result as a new “E” column
    info_ALLState_df["E"] = info_ALLState_df.apply(lambda row: ELA_Network_functions.calc_E_1state(state_str=row["state"], h_1darray=h_1darray, J_2darray=J_2darray), axis=1)
    # return
    return info_ALLState_df