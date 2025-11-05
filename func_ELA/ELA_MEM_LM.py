#https://peluigi.hatenablog.com/entry/2018/06/26/232846
#https://sites.google.com/site/ezakitakahiro/software
#https://tezk.hatenablog.com/entry/2017/07/10/144941
#https://github.com/tkEzaki/energy-landscape-analysis
#https://royalsocietypublishing.org/doi/10.1098/rsta.2016.0287
#https://qiita.com/sci_Haru/items/1ad3b246a2c931a9833d
import numpy as np
import pandas as pd

from typing import Tuple

#-----------------------------------------------------------------------------------
# check if datas contains 0 or 1 
#-----------------------------------------------------------------------------------
def check_if_datas_contains_0or1(binarizedData_ndarray:np.ndarray) -> None:
    #-----------
    #check pd.Dataframe
    if (type(binarizedData_ndarray) is np.ndarray)==False:
        raise ValueError("binarizedData must be np.ndarray !!!")
    #-----------
    #check 0or1
    if np.unique(binarizedData_ndarray).tolist() not in ([0],[1],[0,1]):
        raise ValueError("values of binarizedData must be 0 or 1 !!!")


#-----------------------------------------------------------------------------------
# return ndarray of All state (=2^N pattern)
#-----------------------------------------------------------------------------------
def return_Allstate(n_feature:int) -> np.ndarray:
    #make df of ALLstate
    df_ALLstate = pd.DataFrame(data=[str(format(n,"b")).zfill(n_feature) for n in range(2**n_feature)], columns=["state"])
    #change_to_column
    def func_change_to_column(row):
        for col_index in range(len(row["state"])):
            row[str(col_index)] = int(row["state"][col_index]) #[col_index:col_index+1]
        return row
    df_ALLstate = df_ALLstate.apply(func_change_to_column, axis=1)
    #
    df_ALLstate = df_ALLstate.drop(["state"],axis=1)
    #return
    return df_ALLstate.values

def return_Allstate_list(n_feature:int) -> np.ndarray:
    #return
    return [str(format(n,"b")).zfill(n_feature) for n in range(2**n_feature)]

#-----------------------------------------------------------------------------------
# calc mean_model and corr_model
#-----------------------------------------------------------------------------------
def return_mean_corr_model(h_1darray:np.ndarray, J_2darray:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #-----------
    #var
    n_feature = len(h_1darray)
    ALLstate_ndarray = return_Allstate(n_feature=n_feature)
    #-----------
    #E
    E = -np.dot(h_1darray, ALLstate_ndarray.T) - np.diag(np.dot(ALLstate_ndarray, np.dot(J_2darray, ALLstate_ndarray.T)))/2
    #exp(-E) :len(exp_E)=2^n_feature
    exp_E_1darray = np.exp(-E)
    #Z
    Z_int = sum(exp_E_1darray) #Z = sum(np.exp(-np.dot(h_1darray, ALLstate_ndarray.T).flatten() - np.array([np.dot(s_ndarray.T, np.dot(J_2darray,s_ndarray)) for _, s_ndarray in enumerate(ALLstate_ndarray)])/2 ))
    #P
    P_1darray = exp_E_1darray/Z_int #np.sum(P_model_1darray)=1
    #-----------
    #mean_model, corr_model
    mean_model = np.sum((np.array([P_1darray for _ in range(n_feature)])*ALLstate_ndarray.T), axis=1)
    corr_model = np.dot(ALLstate_ndarray.T, ALLstate_ndarray*(np.array([P_1darray for _ in range(n_feature)]).T))
    #print((np.array([P_1darray for _ in range(n_feature)]).T))
    #print(ALLstate_ndarray*(np.array([P_1darray for _ in range(n_feature)]).T))
    #print(mean_model)
    #print(np.array([P_1darray for _ in range(n_feature)]).T)
    #print(ALLstate_ndarray*(np.array([P_1darray for _ in range(n_feature)]).T))
    #print(corr_model)
    #-----------
    #return
    return mean_model, corr_model

#-----------------------------------------------------------------------------------
# calc h and J
#-----------------------------------------------------------------------------------
from numpy import linalg
from scipy.spatial.distance import squareform
def calc_MEM_LikelihoodMaximization(binarizedData_ndarray:np.ndarray, ipsilon:float=1, permissible_Error:float=0.000001, 
                                    save_path_h_and_J:str=False, save_path_h:str=False, save_path_J:str=False,
                                    print_progress:str=False, save_progress_of_estimation:list=False) -> Tuple[np.ndarray, np.ndarray]:
    #-----------
    #check data
    check_if_datas_contains_0or1(binarizedData_ndarray=binarizedData_ndarray)
    #-----------
    #n_data, n_feature
    n_data = binarizedData_ndarray.shape[0]
    n_feature = binarizedData_ndarray.shape[1]
    #-----------
    #progress_of_estimation
    progress_of_estimation_h = []
    progress_of_estimation_J = []
    progress_of_estimation_error = []
    #-----------
    #var
    iter_max = 50000000
    permissible_Error = permissible_Error #0.00000001
    ipsilon = ipsilon
    #-----------
    #mean_data, corr_data
    mean_data = np.mean(a=binarizedData_ndarray.T, axis=1)
    corr_data = np.dot(binarizedData_ndarray.T,binarizedData_ndarray)/n_data
    #h, J (default)
    h = np.zeros(n_feature)
    J = np.zeros((n_feature,n_feature))
    #-----------
    #calc MEM by LM
    for _ in range(iter_max):
        #-----------
        #mean_model, corr_model
        mean_model, corr_model = return_mean_corr_model(h_1darray=h, J_2darray=J)
        #-----------
        #dh, dJ
        dh = n_data*(mean_data - mean_model) #(ipsilon/n_data)*(mean_data - mean_model)
        dJ = n_data*(corr_data - corr_model) #(ipsilon/n_data)*(corr_data - corr_model)
        #set diag to 0
        dJ = dJ - np.diag(np.diag(dJ))
        #-----------
        #check Error
        #https://qiita.com/sci_Haru/items/1ad3b246a2c931a9833d
        error = np.sqrt(linalg.norm(dJ,2)**2 + linalg.norm(dh,2)**2)/(n_feature*(n_feature+1))
        if print_progress != False:
            print(error)
        if save_progress_of_estimation != False:
            progress_of_estimation_h.append(h)
            progress_of_estimation_J.append(squareform(np.tril(J).T + np.tril(J)))
            progress_of_estimation_error.append(error)
        if error < permissible_Error:
            break
        #-----------
        #update
        h = h + ipsilon*dh #h = h + (ipsilon/n_data)*dh
        J = J + ipsilon*dJ #J = J + (ipsilon/n_data)*dJ
    #-----------
    #save
    if save_path_h_and_J != False:
        #change J to condensed_distance_matrix
        J_condensed_distance_matrix = squareform(np.tril(J).T + np.tril(J)) #squareform(J)はfloatの微妙な違いがあるためエラー
        #concatenate h and J
        h_and_J_ndarray = np.concatenate([h, J_condensed_distance_matrix], 0)
        #make df
        h_and_J_df = pd.DataFrame(data=[h_and_J_ndarray], columns=["h"+str(cnt_1) for cnt_1 in range(n_feature)] + ["J"+str(cnt_1)+str(cnt_2) for cnt_1 in range(n_feature) for cnt_2 in range(n_feature) if cnt_1 < cnt_2])
        #write
        h_and_J_df.to_csv(save_path_h_and_J, header=True, index=False, encoding="utf-8")
    if save_path_h != False:
        #make df
        h_df = pd.DataFrame(data=h, columns=["h"])
        #write
        h_df.to_csv(save_path_h, header=True, index=False, encoding="utf-8")
    if save_path_J != False:
        #make df
        J_df = pd.DataFrame(data=J, columns=["J{}".format(j) for j in range(len(J))])
        #write
        J_df.to_csv(save_path_J, header=True, index=False, encoding="utf-8")
    if save_progress_of_estimation != False:
        #-----------
        #make df
        h_prog_df = pd.DataFrame(data=progress_of_estimation_h, columns=["h"+str(cnt_1) for cnt_1 in range(n_feature)])
        J_prog_df = pd.DataFrame(data=progress_of_estimation_J, columns=["J"+str(cnt_1)+str(cnt_2) for cnt_1 in range(n_feature) for cnt_2 in range(n_feature) if cnt_1 < cnt_2])
        error_prog_df = pd.DataFrame(data={"error":progress_of_estimation_error})
        #-----------
        #set index name
        h_prog_df.index.name = "Iter"
        J_prog_df.index.name = "Iter"
        error_prog_df.index.name = "Iter"
        #-----------
        #write
        h_prog_df.to_csv(save_progress_of_estimation[0], header=True, index=True, encoding="utf-8")
        J_prog_df.to_csv(save_progress_of_estimation[1], header=True, index=True, encoding="utf-8")
        error_prog_df.to_csv(save_progress_of_estimation[2], header=True, index=True, encoding="utf-8")
    #-----------
    #return
    return h, J