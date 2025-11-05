import numpy as np
import pandas as pd

from typing import Tuple, Union

from . import ELA_MEM_LM, ELA_Network_calc_E, ELA_Network_calc_SS, ELA_Network_calc_TP_onlySS, ELA_Network_functions

#-----------------------------------------------------------------------------------
# ⑤calc major local minimum (major SS)
#-----------------------------------------------------------------------------------
#-------------------------
def calc_random_maximum_length_of_branch(t_max:int, N:int, ipsilon:float, permissible_Error:float, n_repeat:int=100, path_save:str=False): # t_max:data length, N:number of features, n_repeat:number of calculation repetitions
    #-----------------
    # random_maximum_length_of_branch_list
    #-----------------
    random_maximum_length_of_branch_list = []

    for cnt in range(n_repeat):
        #-----------------
        # make random data
        df_random = np.random.uniform(0, 1, (t_max, N))
        df_random = np.where(df_random<0.5, 0, 1)
        df_random = pd.DataFrame(data=df_random)
        df_random
        #-----------------
        # calc MEM
        h, J = ELA_MEM_LM.calc_MEM_LikelihoodMaximization(binarizedData_ndarray=df_random.values, 
                                                          ipsilon=ipsilon,
                                                          permissible_Error=permissible_Error,
                                                            save_path_h_and_J=False, save_path_h=False, save_path_J=False,
                                                            print_progress=False)
        #-----------------
        # make Network
        info_ALLState_df = ELA_Network_calc_E.calc_E_ALLstate_from_h_and_J(h_1darray=h, J_2darray=J)
        info_ALLState_df, info_SS_df = ELA_Network_calc_SS.calc_SS(info_ALLState_df=info_ALLState_df, path_save_info_SS=False)
        TP_onlySS_df, TP_onlySS_name_df = ELA_Network_calc_TP_onlySS.calc_TP_onlySS(info_ALLState_df=info_ALLState_df, diag_value="E", 
                                                                    path_save_TP_onlySS=False, path_save_TP_onlySS_name=False,
                                                                    print_progress=False)

        #-----------------
        # Subtract each E value to obtain "energy_barrier" from energy_threshold: the energy barrier viewed from each "columns" state
        TP_onlySS_df_energy_barrier = TP_onlySS_df.copy()
        for index, row_series in TP_onlySS_df.iterrows():
            state_use = index
            E_use = info_SS_df.query('state==@state_use')["E"].values[0]
            TP_onlySS_df_energy_barrier[state_use] = TP_onlySS_df_energy_barrier[state_use] - E_use

        #-----------------
        # For each row, find the minimum value (= branch_length), then take the maximum of those minima (maximum_length_of_branch).
        # Note: The diagonal elements (E=0) represent self-comparisons and should be ignored.
        if 1 < len(TP_onlySS_df_energy_barrier.columns):
            min_branch_length_list = []
            for cnt_col, state_use in enumerate(TP_onlySS_df_energy_barrier.columns):
                min_branch_length = min(TP_onlySS_df_energy_barrier.drop(index=state_use)[state_use])
                min_branch_length_list.append(min_branch_length)
            max_branch_length = max(min_branch_length_list)
        elif len(TP_onlySS_df_energy_barrier.columns) == 1:
            max_branch_length = 0
        else:
            raise ValueError("error!!!!!")

        #-----------------
        # append to result list
        random_maximum_length_of_branch_list.append(max_branch_length)


    #-----------------
    # calculate mean and std
    random_maximum_length_of_branch_mean = np.mean(np.array(random_maximum_length_of_branch_list))
    random_maximum_length_of_branch_std = np.std(np.array(random_maximum_length_of_branch_list))

    #-----------------
    # write
    if path_save != False:
        df_write = pd.DataFrame(data=[[t_max, N, n_repeat, random_maximum_length_of_branch_mean, random_maximum_length_of_branch_std]], 
                                columns=["t_max", "N", "n_repeat", "random_maximum_length_of_branch_mean", "random_maximum_length_of_branch_std"])
        df_write.to_csv(path_save, header=True, index=False, encoding="utf-8")


    #-----------------
    # return
    return random_maximum_length_of_branch_mean, random_maximum_length_of_branch_std


#-------------------------
def calc_major_local_minimum(N:int, t_max:int, n_repeat:int,  # t_max:data length, N:number of features, n_repeat:number of calculation repetitions
                             ipsilon:float, permissible_Error:float,
                             info_ALLState_df:pd.DataFrame, info_SS_df:pd.DataFrame, TP_onlySS_df:pd.DataFrame, TP_onlySS_name_df:pd.DataFrame,
                             path_random_maximum_length_of_branch:str=False, 
                             path_info_ALLState_majorSS:str=False, path_info_SS_majorSS:str=False, path_TP_onlySS_majorSS:str=False, path_TP_onlySS_name_majorSS:str=False):
    #-----------------
    # calc mean and std of random maximum length of branch
    random_maximum_length_of_branch_mean, random_maximum_length_of_branch_std = calc_random_maximum_length_of_branch(t_max=t_max, N=N, 
                                                                                                                     ipsilon=ipsilon, permissible_Error=permissible_Error, 
                                                                                                                     n_repeat=n_repeat, path_save=path_random_maximum_length_of_branch)
    
    #-----------------
    # Subtract each E value to obtain "energy_barrier" from energy_threshold: the energy barrier viewed from each "columns" state
    TP_onlySS_df_energy_barrier = TP_onlySS_df.copy()
    for index, row_series in TP_onlySS_df.iterrows():
        state_use = index
        E_use = info_SS_df.query('state==@state_use')["E"].values[0]
        TP_onlySS_df_energy_barrier[state_use] = TP_onlySS_df_energy_barrier[state_use] - E_use

    #-----------------
    # Fill the diagonal of TP_onlySS_df with inf
    for cnt_col, state_use in enumerate(TP_onlySS_df_energy_barrier.columns):
        TP_onlySS_df_energy_barrier.at[state_use,state_use] = np.inf

    #-----------------
    # calc
    while True:
        #-----------------
        # min_branch_length (=min of "min for each cols")
        min_state = ""
        min_branch_length = np.inf
        for cnt_col, state_use in enumerate(TP_onlySS_df_energy_barrier.columns):
            # Minimum value of the current column
            min_branch_length_this_col = TP_onlySS_df_energy_barrier[state_use].values.min()
            # check
            if min_branch_length_this_col <= min_branch_length:
                # update
                min_state =state_use
                min_branch_length = min_branch_length_this_col
        #-----------------
        # thresholds
        thresholds = random_maximum_length_of_branch_mean + 2*random_maximum_length_of_branch_std
        #-----------------
        # If the min_branch_length is smaller than the threshold, cut the corresponding row and column
        if min_branch_length<thresholds:
            TP_onlySS_df_energy_barrier = TP_onlySS_df_energy_barrier.drop(min_state, axis=0) # drop index
            TP_onlySS_df_energy_barrier = TP_onlySS_df_energy_barrier.drop(min_state, axis=1) # drop column
        # If the min_branch_length is greater than the threshold, break the process
        else :
            break

    #-----------------
    # Remaining SS
    SS_major = TP_onlySS_df_energy_barrier.columns.to_list()

    #-----------------
    # make dfs & write
    #--------
    # info_ALLState_majorSS_df
    def func(row):
        if row["state"] in SS_major:
            row["majorSS"] = True
        else:
            row["majorSS"] = False
        return row
    info_ALLState_majorSS_df = info_ALLState_df.apply(func, axis=1)
    # write
    if path_info_ALLState_majorSS != False:
        info_ALLState_majorSS_df.to_csv(path_info_ALLState_majorSS, header=True, index=False, encoding="utf-8")

    #--------
    # info_SS_majorSS_df
    info_SS_majorSS_df = info_ALLState_majorSS_df.query('majorSS == True')
    # write
    if path_info_SS_majorSS != False:
        info_SS_majorSS_df.to_csv(path_info_SS_majorSS, header=True, index=False, encoding="utf-8")

    #--------
    # TP_onlySS_majorSS_df
    TP_onlySS_majorSS_df = TP_onlySS_df.loc[SS_major,SS_major]
    # write
    if path_TP_onlySS_majorSS != False:
        TP_onlySS_majorSS_df.to_csv(path_TP_onlySS_majorSS, header=True, index=True, encoding="utf-8")

    #--------
    # TP_onlySS_name_majorSS_df
    TP_onlySS_name_majorSS_df = TP_onlySS_name_df.loc[SS_major,SS_major]
    # write
    if path_TP_onlySS_name_majorSS != False:
        TP_onlySS_name_majorSS_df.to_csv(path_TP_onlySS_name_majorSS, header=True, index=True, encoding="utf-8")

    #-----------------
    # return
    return info_ALLState_majorSS_df, info_SS_majorSS_df, TP_onlySS_majorSS_df, TP_onlySS_name_majorSS_df
