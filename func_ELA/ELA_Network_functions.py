import numpy as np
import pandas as pd

#-----------------------------------------------------------------------------------
#-----------------------------
# func: input:state -> output:List of adjacent states (AS) of the given state
def return_AS_binary(state_str :str) -> list:
    #-------------------------
    # check 0 or 1
    unique_var_list = list(set(list(state_str)))
    for _, unique_var in enumerate(unique_var_list):
        if unique_var != "0" and unique_var != "1":
            raise ValueError("state must be 0 or 1")
    #-------------------------
    # make Adjacent States
    AS_list = []
    for cnt, var in enumerate(state_str):
        # var_reversal
        var_reversal = "1" if var=="0" else "0"
        # AS
        AS = state_str[:cnt]+var_reversal+state_str[cnt+1:]
        # append
        AS_list.append(AS)
    return AS_list

#-----------------------------
# func: check the shape of h & J
def check_shape(h_1darray:np.ndarray, J_2darray:np.ndarray) -> None:
    # check dim
    if not(h_1darray.ndim == 1 and J_2darray.ndim == 2):
        raise ValueError("dim erorr")
    # check shape of J_2darray
    if J_2darray.shape[0] != J_2darray.shape[1]:
        raise ValueError("shape of J_2darray erorr")
    # check len
    if h_1darray.shape[0] != J_2darray.shape[0]:
        raise ValueError("len erorr")

#-----------------------------
# func: input:h & J -> output:DataFrame of all states(ALLstate_df)
def make_ALLstate(h_1darray:np.ndarray, J_2darray:np.ndarray) -> pd.DataFrame:
    # len of state
    len_state = h_1darray.shape[0]
    # make df of ALLstate
    ALLstate_df = pd.DataFrame(data=[str(format(n,"b")).zfill(len_state) for n in range(2**len_state)], columns=["state"])
    return ALLstate_df

#-----------------------------
# func: input:state, h, J -> output:E
def calc_E_1state(state_str:str, h_1darray:np.array, J_2darray:np.array) -> float:
    # state_str -> state_1darray
    state_1darray = np.array([int(s) for _, s in enumerate(state_str)])
    # calc E
    E = -np.dot(h_1darray,state_1darray) - np.sum(np.multiply(J_2darray,np.outer(state_1darray,state_1darray)))/2
    return E



#-----------------------------------------------------------------------------------
#-----------------------------
def make_color_dict(path_read_info_ALLState:str) -> dict:
    #----------------------
    # read
    info_ALLState_df = pd.read_csv(path_read_info_ALLState, index_col=None, header=0, sep=',', encoding="utf-8", dtype={"state":str,"E":float,"SS":bool,"RA":str,"next_state":str})
    #----------------------
    # SS_color_dict
    color_valuelist = ["#ea5532","#006a6c","#47266e","#d70035","#003f8e","#6c3524","#7e837f","#e5a323","#003f8e","#000000","#5f6527","#e8c59c","#e95464","#82cddd","#a7d28d","#8da0b6","#98605e","#9cbb1c","#e44d93","#006400","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""]
    SS_list = info_ALLState_df.query('SS==True')["state"].to_list()
    SS_color_dict = dict(zip(SS_list, color_valuelist[:len(SS_list)]))
    #----------------------
    # return
    return SS_color_dict


#-----------------------------------------------------------------------------------
# generate the color code of gradation color from int (base_color -> white)
#-----------------------------
def get_color_of_gradation_from_int_to_white(base_color: str, use_num: float, min:float, max:float) -> str:
    # check vars
    if len(base_color) != 7:
        raise ValueError('Color must be #------')
    if use_num < min:
        raise ValueError('min must be smaller than use_num')
    if max < use_num:
        raise ValueError('max must be larger than use_num')
    # change color code to RGB
    r = int(base_color[1:3], 16)
    g = int(base_color[3:5], 16)
    b = int(base_color[5:7], 16)
    # make RBG of gradation
    r_gradation = int((255 - r) * ((use_num-min) / (max-min)) + r)
    g_gradation = int((255 - g) * ((use_num-min) / (max-min)) + g)
    b_gradation = int((255 - b) * ((use_num-min) / (max-min)) + b)
    # return the color code of gradation
    return f'#{r_gradation:02x}{g_gradation:02x}{b_gradation:02x}'

#-----------------------------
def get_color_of_gradation_from_int_from_white(base_color: str, use_num: float, min:float, max:float) -> str:
    # check vars
    if len(base_color) != 7:
        raise ValueError('Color must be #------')
    if use_num < min:
        raise ValueError('min must be smaller than use_num')
    if max < use_num:
        raise ValueError('max must be larger than use_num')
    # change color code to RGB
    r = int(base_color[1:3], 16)
    g = int(base_color[3:5], 16)
    b = int(base_color[5:7], 16)
    # make RBG of gradation
    r_gradation = int((255 - r) * ((max-use_num) / (max-min)) + r)
    g_gradation = int((255 - g) * ((max-use_num) / (max-min)) + g)
    b_gradation = int((255 - b) * ((max-use_num) / (max-min)) + b)
    # return the color code of gradation
    return f'#{r_gradation:02x}{g_gradation:02x}{b_gradation:02x}'

#-----------------------------
def get_color_of_gradation_from_int_to_black(base_color: str, use_num: float, min:float, max:float) -> str:
    # check vars
    if len(base_color) != 7:
        raise ValueError('Color must be #------')
    if use_num < min:
        raise ValueError('min must be smaller than use_num')
    if max < use_num:
        raise ValueError('max must be larger than use_num')
    # change color code to RGB
    r = int(base_color[1:3], 16)
    g = int(base_color[3:5], 16)
    b = int(base_color[5:7], 16)
    # make RBG of gradation
    r_gradation = int(r * ((max-use_num) / (max-min)))
    g_gradation = int(g * ((max-use_num) / (max-min)))
    b_gradation = int(b * ((max-use_num) / (max-min)))
    # return the color code of gradation
    return f'#{r_gradation:02x}{g_gradation:02x}{b_gradation:02x}'

#-----------------------------
def get_color_of_gradation_from_int_from_black(base_color: str, use_num: float, min:float, max:float) -> str:
    # check vars
    if len(base_color) != 7:
        raise ValueError('Color must be #------')
    if use_num < min:
        raise ValueError('min must be smaller than use_num')
    if max < use_num:
        raise ValueError('max must be larger than use_num')
    # change color code to RGB
    r = int(base_color[1:3], 16)
    g = int(base_color[3:5], 16)
    b = int(base_color[5:7], 16)
    # make RBG of gradation
    r_gradation = int(r * ((use_num-min) / (max-min)))
    g_gradation = int(g * ((use_num-min) / (max-min)))
    b_gradation = int(b * ((use_num-min) / (max-min)))
    # return the color code of gradation
    return f'#{r_gradation:02x}{g_gradation:02x}{b_gradation:02x}'



