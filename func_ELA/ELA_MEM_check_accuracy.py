#https://royalsocietypublishing.org/doi/10.1098/rsta.2016.0287
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple

from . import ELA_MEM_LM

#-----------------------------------------------------------------------------------
# plot_model_empirical
#-----------------------------------------------------------------------------------
#-----------
def plot_model_empirical(binarizedData_df:pd.DataFrame, h_1darray:np.ndarray, J_2darray:np.ndarray, 
                         log_scale:bool,
                         lim:float, path_save:str=False) -> None:
    #-----------
    #r, I2/IN
    #-----------
    #r
    r = return_r(binarizedData_df=binarizedData_df, h_1darray=h_1darray, J_2darray=J_2darray)
    #I2_IN
    I2_IN = return_I2_IN(binarizedData_df=binarizedData_df, h_1darray=h_1darray, J_2darray=J_2darray)
    #-----------
    #preprocessing
    #-----------
    #-----------
    #binarizedData_dfを全てintにする
    binarizedData_df = binarizedData_df.astype('int')
    #-----------
    #datas_df (stateを0,1,1 -> 011)
    datas_df = binarizedData_df
    datas_df["state"] = datas_df.apply(lambda row: "".join([str(v) for _, v in enumerate(row)]), axis=1)
    #-----------
    #var
    n_feature = len(h_1darray)
    df_ALLstate = pd.DataFrame(data=[str(format(n,"b")).zfill(n_feature) for n in range(2**n_feature)], columns=["state"])
    #-----------
    #P_empirical
    #-----------
    def func_calc_P(row):
        #state_str
        state_str = row["state"]
        #P_empirical
        row["P_empirical"] = len(datas_df.query('state==@state_str'))/len(datas_df)
        #return
        return row
    df_ALLstate = df_ALLstate.apply(func_calc_P, axis=1)
    #-----------
    #P_model
    #-----------
    #ALLstate_ndarray
    ALLstate_ndarray = ELA_MEM_LM.return_Allstate(n_feature=n_feature)
    #E
    E = -np.dot(h_1darray, ALLstate_ndarray.T) - np.diag(np.dot(ALLstate_ndarray, np.dot(J_2darray, ALLstate_ndarray.T)))/2
    #exp(-E) :len(exp_E)=2^n_feature
    exp_E_1darray = np.exp(-E)
    #Z
    Z_int = sum(exp_E_1darray) #Z = sum(np.exp(-np.dot(h_1darray, ALLstate_ndarray.T).flatten() - np.array([np.dot(s_ndarray.T, np.dot(J_2darray,s_ndarray)) for _, s_ndarray in enumerate(ALLstate_ndarray)])/2 ))
    #P
    P_1darray = exp_E_1darray/Z_int #np.sum(P_model_1darray)=1
    #P_model
    df_ALLstate["P_model"] = P_1darray
    #-----------
    #print
    print(df_ALLstate)
    print("P_empirical",df_ALLstate["P_empirical"].sum())
    print("P_model",df_ALLstate["P_model"].sum())
    #-----------
    #plot 誤差 train (x:P_empirical y:P_model)
    #-----------
    #x_list, y_list
    x_list = df_ALLstate["P_empirical"].tolist()
    y_list = df_ALLstate["P_model"].tolist()
    #fig, ax
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1,1,1)
    #plot scatter
    ax.scatter(x=x_list, y=y_list, s=80, c="black")
    #plot y=x
    #x=np.linspace(0, max(x_list+y_list), 15) # 直線をプロット
    x=np.linspace(0, lim, 100) # 直線をプロット
    y=x
    ax.plot(x, y, color = "black")
    #plot r I2/IN
    plt.text(x=0.99, y=3*0.1, s="$n = {}$".format(len(binarizedData_df)), 
                         fontdict=dict(fontsize=40,fontstyle="italic"), va='top', ha='right', transform=ax.transAxes)
    plt.text(x=0.99, y=2*0.1, s="$r = {}$".format("{:.5f}".format(r)), 
                         fontdict=dict(fontsize=40,fontstyle="italic"), va='top', ha='right', transform=ax.transAxes)
    plt.text(x=0.99, y=1*0.1, s="$I_2/I_N = {}$".format("{:.5f}".format(I2_IN)), 
                         fontdict=dict(fontsize=40,fontstyle="italic"), va='top', ha='right', transform=ax.transAxes)
    #setting
    ax.tick_params(labelsize = 30)#軸の大きさ
    ax.set_xlabel("P_data",fontsize=50) #P_empirical
    ax.set_ylabel("P_model",fontsize=50)
    #plt.grid()
    #log_scale
    if log_scale==True:
        plt.xscale('log')
        plt.yscale('log')
    else:
        padding=0.003
        ax.set_xlim([-padding,lim+padding]) #ax.set_xlim([-padding,max(x_list+y_list)+padding])
        ax.set_ylim([-padding,lim+padding]) #ax.set_ylim([-padding,max(x_list+y_list)+padding])
    #save
    if path_save != False:
        plt.savefig(path_save, bbox_inches="tight")
    plt.show()
#test
#n_state = 5
#binarizedData_df = pd.DataFrame(data=np.random.randint(0, 2, (300,n_state)))
#h_1darray=np.random.rand(n_state)
#J_2darray=np.random.normal(-1, 0.5, (n_state, n_state))

#plot_model_empirical(binarizedData_df=binarizedData_df, h_1darray=h_1darray, J_2darray=J_2darray)


#-----------
def plot_model_empirical_from_path(binarizedData_df:pd.DataFrame, h_path:str, J_path:str, 
                                   log_scale:bool,
                                   lim:float, path_save:str=False) -> None:
    #-----------
    #load h_1darray, J_2darray
    h_1darray = pd.read_csv(h_path, index_col=None, header=0, sep=',').values.flatten()
    J_2darray = pd.read_csv(J_path, index_col=None, header=0, sep=',').values
    #-----------
    #plot_model_empirical
    plot_model_empirical(binarizedData_df=binarizedData_df, h_1darray=h_1darray, J_2darray=J_2darray, 
                         log_scale=log_scale, lim=lim, path_save=path_save)


#-----------------------------------------------------------------------------------
# I2_IN
#-----------------------------------------------------------------------------------
#-----------
#ShannonEntropy
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
from scipy.stats import entropy

def return_ShannonEntropy(Px_1darray:np.ndarray) -> int: #Px_1darray,Qx_1darrayは確率 ex.[3/5, 1/5, 1/5]
    return entropy(Px_1darray)
#test 
#p = np.array([5/30, 10/30, 2/30, 3/30, 5/30, 5/30]) #dist p
#q = np.array([0, 0, 0, 1/2, 1/2,0]) #dist q
#return_ShannonEntropy(Px_1darray=q) #calc ShannonEntropy

#-----------
def return_I2_IN(binarizedData_df:pd.DataFrame, h_1darray:np.ndarray, J_2darray:np.ndarray) -> int:
    #-----------
    #datas_df (stateを0,1,1 -> 011)
    datas_df = binarizedData_df.astype('int')
    datas_df["state"] = datas_df.apply(lambda row: "".join([str(v) for _, v in enumerate(row)]), axis=1)
    print(datas_df)
    #-----------
    #var
    n_feature = len(h_1darray)
    df_ALLstate = pd.DataFrame(data=[str(format(n,"b")).zfill(n_feature) for n in range(2**n_feature)], columns=["state"])
    #-----------
    #P_1
    #-----------
    #ALLstate_ndarray
    ALLstate_ndarray = ELA_MEM_LM.return_Allstate(n_feature=n_feature)
    #E
    E = -np.dot(h_1darray, ALLstate_ndarray.T)/2
    #exp(-E) :len(exp_E)=2^n_feature
    exp_E_1darray = np.exp(-E)
    #Z
    Z_int = sum(exp_E_1darray) #Z = sum(np.exp(-np.dot(h_1darray, ALLstate_ndarray.T).flatten() - np.array([np.dot(s_ndarray.T, np.dot(J_2darray,s_ndarray)) for _, s_ndarray in enumerate(ALLstate_ndarray)])/2 ))
    #P
    P_1darray = exp_E_1darray/Z_int #np.sum(P_model_1darray)=1
    #P_model
    df_ALLstate["P_1"] = P_1darray
    #-----------
    #P_2 (P_model)
    #-----------
    #ALLstate_ndarray
    ALLstate_ndarray = ELA_MEM_LM.return_Allstate(n_feature=n_feature)
    #E
    E = -np.dot(h_1darray, ALLstate_ndarray.T) - np.diag(np.dot(ALLstate_ndarray, np.dot(J_2darray, ALLstate_ndarray.T)))/2
    #exp(-E) :len(exp_E)=2^n_feature
    exp_E_1darray = np.exp(-E)
    #Z
    Z_int = sum(exp_E_1darray) #Z = sum(np.exp(-np.dot(h_1darray, ALLstate_ndarray.T).flatten() - np.array([np.dot(s_ndarray.T, np.dot(J_2darray,s_ndarray)) for _, s_ndarray in enumerate(ALLstate_ndarray)])/2 ))
    #P
    P_1darray = exp_E_1darray/Z_int #np.sum(P_model_1darray)=1
    #P_model
    df_ALLstate["P_2"] = P_1darray
    #-----------
    #P_N (P_empirical)
    #-----------
    def func_calc_P(row):
        #state_str
        state_str = row["state"]
        #P_empirical
        row["P_N"] = len(datas_df.query('state==@state_str'))/len(datas_df)
        #return
        return row
    df_ALLstate = df_ALLstate.apply(func_calc_P, axis=1)
    #-----------
    #print
    print(df_ALLstate)
    print("P_1",df_ALLstate["P_1"].sum())
    print("P_2",df_ALLstate["P_2"].sum())
    print("P_N",df_ALLstate["P_N"].sum())

    #-----------
    #calc I2_IN
    #-----------
    #P_1, P_2, P_N
    P_1 = df_ALLstate["P_1"].values
    P_2 = df_ALLstate["P_2"].values
    P_N = df_ALLstate["P_N"].values
    #I2_IN
    S_1 = return_ShannonEntropy(Px_1darray=P_1)
    S_2 = return_ShannonEntropy(Px_1darray=P_2)
    S_N = return_ShannonEntropy(Px_1darray=P_N)
    I2_IN = (S_1 - S_2)/(S_1 - S_N)
    #-----------
    #return
    return I2_IN



#-----------------------------------------------------------------------------------
# r
#-----------------------------------------------------------------------------------
#-----------
#KL_divergence (relative entropy)
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html#scipy.special.rel_entr
#https://stackoverflow.com/questions/63369974/3-functions-for-computing-relative-entropy-in-scipy-whats-the-difference
"""from scipy.stats import entropy
def return_KL_div(Px_1darray:np.ndarray, Qx_1darray:np.ndarray) -> int: #Px_1darray,Qx_1darrayは確率 ex.[3/5, 1/5, 1/5]
    return entropy(Px_1darray, Qx_1darray)"""

def return_KL_div(Px_1darray:np.ndarray, Qx_1darray:np.ndarray) -> int: #Px_1darray,Qx_1darrayは確率 ex.Px_1darray=[3/5, 1/5, 1/5]
    KL_div = 0
    for p, q in zip(Px_1darray,Qx_1darray):
        if p<0 or q<0:
            raise ValueError("error!!!!!")
        elif q==0:
            KL_div += 0
            #raise ValueError("error!!!!!")
        elif p==0:
            KL_div += 0
        else: #p>0 and q>0
            KL_div += p*np.log(p/q)
    return KL_div
#test 
#p = np.array([2/10, 3/10, 1/10, 3/10, 1/10]) #dist p
#q = np.array([1/10, 1/10, 1/10, 4/10, 3/10]) #dist q
#print(return_KL_div(Px_1darray=p, Qx_1darray=q)) #calc kL_div
#print(return_KL_div(Px_1darray=q, Qx_1darray=p)) #calc kL_div

#-----------
def return_r(binarizedData_df:pd.DataFrame, h_1darray:np.ndarray, J_2darray:np.ndarray) -> int:
    #-----------
    #datas_df (stateを0,1,1 -> 011)
    datas_df = binarizedData_df.astype('int')
    datas_df["state"] = datas_df.apply(lambda row: "".join([str(v) for _, v in enumerate(row)]), axis=1)
    #print(datas_df)
    #-----------
    #var
    n_feature = len(h_1darray)
    df_ALLstate = pd.DataFrame(data=[str(format(n,"b")).zfill(n_feature) for n in range(2**n_feature)], columns=["state"])
    #-----------
    #P_1
    #-----------
    #ALLstate_ndarray
    ALLstate_ndarray = ELA_MEM_LM.return_Allstate(n_feature=n_feature)
    #E
    E = -np.dot(h_1darray, ALLstate_ndarray.T)/2
    #exp(-E) :len(exp_E)=2^n_feature
    exp_E_1darray = np.exp(-E)
    #Z
    Z_int = sum(exp_E_1darray) #Z = sum(np.exp(-np.dot(h_1darray, ALLstate_ndarray.T).flatten() - np.array([np.dot(s_ndarray.T, np.dot(J_2darray,s_ndarray)) for _, s_ndarray in enumerate(ALLstate_ndarray)])/2 ))
    #P
    P_1darray = exp_E_1darray/Z_int #np.sum(P_model_1darray)=1
    #P_model
    df_ALLstate["P_1"] = P_1darray
    #-----------
    #P_2 (P_model)
    #-----------
    #ALLstate_ndarray
    ALLstate_ndarray = ELA_MEM_LM.return_Allstate(n_feature=n_feature)
    #E
    E = -np.dot(h_1darray, ALLstate_ndarray.T) - np.diag(np.dot(ALLstate_ndarray, np.dot(J_2darray, ALLstate_ndarray.T)))/2
    #exp(-E) :len(exp_E)=2^n_feature
    exp_E_1darray = np.exp(-E)
    #Z
    Z_int = sum(exp_E_1darray) #Z = sum(np.exp(-np.dot(h_1darray, ALLstate_ndarray.T).flatten() - np.array([np.dot(s_ndarray.T, np.dot(J_2darray,s_ndarray)) for _, s_ndarray in enumerate(ALLstate_ndarray)])/2 ))
    #P
    P_1darray = exp_E_1darray/Z_int #np.sum(P_model_1darray)=1
    #P_model
    df_ALLstate["P_2"] = P_1darray
    #-----------
    #P_N (P_empirical)
    #-----------
    def func_calc_P(row):
        #state_str
        state_str = row["state"]
        #P_empirical
        row["P_N"] = len(datas_df.query('state==@state_str'))/len(datas_df)
        #return
        return row
    df_ALLstate = df_ALLstate.apply(func_calc_P, axis=1)
    #-----------
    #print
    #print(df_ALLstate)
    #print("P_1",df_ALLstate["P_1"].sum())
    #print("P_2",df_ALLstate["P_2"].sum())
    #print("P_N",df_ALLstate["P_N"].sum())

    #-----------
    #calc r
    #-----------
    #P_1, P_2, P_N
    P_1 = df_ALLstate["P_1"].values
    P_2 = df_ALLstate["P_2"].values
    P_N = df_ALLstate["P_N"].values
    #r
    #r = (return_KL_div(Px_1darray=P_1, Qx_1darray=P_N) - return_KL_div(Px_1darray=P_2, Qx_1darray=P_N)) / return_KL_div(Px_1darray=P_1, Qx_1darray=P_N)
    r = (return_KL_div(Px_1darray=P_N, Qx_1darray=P_1) - return_KL_div(Px_1darray=P_N, Qx_1darray=P_2)) / return_KL_div(Px_1darray=P_N, Qx_1darray=P_1)
    #-----------
    #return
    return r