#!/usr/bin/env python3
# Marcos del Cueto
# Department of Chemistry and MIF, University of Liverpool
#################################################################################
import ast
import sys
import dask
import random
import math
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from time import time
from dask import delayed
from sklearn import neighbors
from numpy.random import choice
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
#################################################################################
######   START CUSTOMIZABLE PARAMETERS ########
input_file_name = 'input_MLL.txt'      # name of input file
######   END CUSTOMIZABLE PARAMETERS   ######
#################################################################################

#################################################################################
###### START MAIN ######
def main(iseed):
    check_input_values()
    results=[]
    if is_dask==True:
        for l in range(Nspf):
            iseed=iseed+l
            provi_result=delayed(MLL)(iseed,l,verbose_level)
            results.append(provi_result)
        results_per_Nspf=dask.compute(results,scheduler='processes',num_workers=NCPU)
        results_per_walker=[list(i) for i in zip(*results_per_Nspf[0])] # transpose
    elif is_dask==False:
        for l in range(Nspf):
            iseed=iseed+l
            provi_result=MLL(iseed,l,verbose_level)
            results.append(provi_result)
        results_per_Nspf=results
        results_per_walker=[list(i) for i in zip(*results_per_Nspf)] # transpose

    print('--- Final results:',flush=True)
    if error_metric == 'rmse':
        for i in range(Nwalkers):
            print('-- Adventurousness: %6.1f' %(adven[i]),flush=True)
            print('- RMSE:',results_per_walker[i][:],flush=True)
            print('- Mean: %f' %(statistics.mean(results_per_walker[i])),flush=True)
    else:
        for i in range(Nspf):
            print('-- Result for landscape %i: %s' %(i,str(results_per_Nspf[0][i])),flush=True)
        print("")
        for i in range(Nwalkers):
            print('-- Adventurousness: %6.1f' %(adven[i]),flush=True)
            print('- Result for Nwalker %i: %s' %(i,str(results_per_walker[i])),flush=True)
            print("")

    if error_metric!=None: plot(error_metric,results_per_walker)
###### END MAIN ######
#################################################################################

#################################################################################
###### START OTHER FUNCTIONS ######
### Function reading input parameters
def read_initial_values(inp):
    # open input file
    input_file_name = inp
    f_in = open('%s' %input_file_name,'r')
    f1 = f_in.readlines()
    # initialize arrays
    input_info = []
    var_name = []
    var_value = []
    # read info before comments. Ignore commented lines and blank lines
    for line in f1:
        if not line.startswith("#") and line.strip(): 
            input_info.append(line.split('#',1)[0].strip())
    # read names and values of variables
    for i in range(len(input_info)):
        var_name.append(input_info[i].split('=')[0].strip())
        var_value.append(input_info[i].split('=')[1].strip())
    # close input file
    f_in.close()
    # assign input variables    
    is_dask = ast.literal_eval(var_value[var_name.index('is_dask')])
    NCPU = ast.literal_eval(var_value[var_name.index('NCPU')])
    verbose_level = ast.literal_eval(var_value[var_name.index('verbose_level')])
    log_name = ast.literal_eval(var_value[var_name.index('log_name')])
    Nspf = ast.literal_eval(var_value[var_name.index('Nspf')])
    S = ast.literal_eval(var_value[var_name.index('S')])
    iseed = ast.literal_eval(var_value[var_name.index('iseed')])
    param = ast.literal_eval(var_value[var_name.index('param')])
    center_min = ast.literal_eval(var_value[var_name.index('center_min')])
    center_max = ast.literal_eval(var_value[var_name.index('center_max')])
    grid_min = ast.literal_eval(var_value[var_name.index('grid_min')])
    grid_max = ast.literal_eval(var_value[var_name.index('grid_max')])
    grid_Delta = ast.literal_eval(var_value[var_name.index('grid_Delta')])
    Nwalkers = ast.literal_eval(var_value[var_name.index('Nwalkers')])
    adven = ast.literal_eval(var_value[var_name.index('adven')])
    t1_time = ast.literal_eval(var_value[var_name.index('t1_time')])
    d_threshold = ast.literal_eval(var_value[var_name.index('d_threshold')])
    t0_time = ast.literal_eval(var_value[var_name.index('t0_time')])
    initial_sampling = ast.literal_eval(var_value[var_name.index('initial_sampling')])
    ML = ast.literal_eval(var_value[var_name.index('ML')])
    error_metric = ast.literal_eval(var_value[var_name.index('error_metric')])
    CV = ast.literal_eval(var_value[var_name.index('CV')])
    k_fold = ast.literal_eval(var_value[var_name.index('k_fold')])
    test_last_percentage = ast.literal_eval(var_value[var_name.index('test_last_percentage')])
    n_neighbor = ast.literal_eval(var_value[var_name.index('n_neighbor')])
    weights = ast.literal_eval(var_value[var_name.index('weights')])
    GBR_criterion = ast.literal_eval(var_value[var_name.index('GBR_criterion')])
    GBR_n_estimators = ast.literal_eval(var_value[var_name.index('GBR_n_estimators')])
    GBR_learning_rate = ast.literal_eval(var_value[var_name.index('GBR_learning_rate')])
    GBR_max_depth = ast.literal_eval(var_value[var_name.index('GBR_max_depth')])
    GBR_min_samples_split = ast.literal_eval(var_value[var_name.index('GBR_min_samples_split')])
    GBR_min_samples_leaf = ast.literal_eval(var_value[var_name.index('GBR_min_samples_leaf')])
    A_RBF = ast.literal_eval(var_value[var_name.index('A_RBF')])
    A_noise = ast.literal_eval(var_value[var_name.index('A_noise')])
    GPR_alpha = ast.literal_eval(var_value[var_name.index('GPR_alpha')])
    kernel_length_scale = ast.literal_eval(var_value[var_name.index('kernel_length_scale')])
    kernel_noise_level = ast.literal_eval(var_value[var_name.index('kernel_noise_level')])
    KRR_alpha = ast.literal_eval(var_value[var_name.index('KRR_alpha')])
    KRR_kernel = ast.literal_eval(var_value[var_name.index('KRR_kernel')])
    KRR_gamma = ast.literal_eval(var_value[var_name.index('KRR_gamma')])
    optimize_gamma = ast.literal_eval(var_value[var_name.index('optimize_gamma')])
    KRR_gamma_lim = ast.literal_eval(var_value[var_name.index('KRR_gamma_lim')])
    allowed_initial_sampling = ast.literal_eval(var_value[var_name.index('allowed_initial_sampling')])
    allowed_CV = ast.literal_eval(var_value[var_name.index('allowed_CV')])
    allowed_ML = ast.literal_eval(var_value[var_name.index('allowed_ML')])
    allowed_error_metric = ast.literal_eval(var_value[var_name.index('allowed_error_metric')])
    t2_time = ast.literal_eval(var_value[var_name.index('t2_time')])
    allowed_verbosity_level = ast.literal_eval(var_value[var_name.index('allowed_verbosity_level')])
    t2_ML = ast.literal_eval(var_value[var_name.index('t2_ML')])
    allowed_t2_ML = ast.literal_eval(var_value[var_name.index('allowed_t2_ML')])

    width_min=S                                     # Minimum width of each Gaussian function
    width_max=1.0/3.0                               # Maximum width of each Gaussian function
    Amplitude_min=0.0                               # Minimum amplitude of each Gaussian function
    Amplitude_max=1.0                               # Maximum amplitude of each Gaussian function
    N=int(round((1/(S**param))))                    # Number of Gaussian functions of a specific landscape
    if iseed==None: 
        iseed=random.randrange(2**30-1) # If no seed is specified, choose a random one

    return (is_dask,NCPU,verbose_level,log_name,Nspf,S,iseed,param,center_min,center_max,grid_min,grid_max,grid_Delta,Nwalkers,adven,t1_time,d_threshold,t0_time,initial_sampling,ML,error_metric,CV,k_fold,test_last_percentage,n_neighbor,weights,GBR_criterion,GBR_n_estimators,GBR_learning_rate,GBR_max_depth,GBR_min_samples_split,GBR_min_samples_leaf,A_RBF,A_noise,GPR_alpha,kernel_length_scale,kernel_noise_level,KRR_alpha,KRR_kernel,KRR_gamma,optimize_gamma,KRR_gamma_lim,allowed_initial_sampling,allowed_CV,allowed_ML,allowed_ML,allowed_error_metric,width_min,width_max,Amplitude_min,Amplitude_max,N,t2_time,allowed_verbosity_level,t2_ML,allowed_t2_ML)

def MLL(iseed,l,verbose_level):
    # open log file to write intermediate information
    if verbose_level>=1:
        f_out = open('%s_%s.log' % (log_name,l), 'w')
    else:
        f_out=None
    error_metric_list=[]
    ML_benefits_list=[]
    dim_list, G_list, Ngrid, max_G = generate_grid(iseed,l,f_out)
    # For each walker
    for w in range(Nwalkers):
        # Step 1) Perform t1 exploration
        X1,y1,unique_t1 = explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,t0_time,t1_time,0,None,None,False)
        if error_metric != None:
        # Step 2A) Calculate error_metric
            if ML=='kNN': error_metric_result=kNN(X1,y1,iseed,l,w,f_out,None,None,1)
            if ML=='GBR': error_metric_result=GBR(X1,y1,iseed,l,w,f_out)
            if ML=='GPR': error_metric_result=GPR(X1,y1,iseed,l,w,f_out,None,None,1)
            if ML=='KRR':
                hyperparams=KRR_gamma
                if optimize_gamma == False:
                    error_metric_result=KRR(hyperparams,X1,y1,iseed,l,w,f_out,verbose_level)
                else:
                    mini_args=(X1,y1,iseed,l,w,f_out,verbose_level)
                    bounds = [KRR_gamma_lim]
                    solver=differential_evolution(KRR,bounds,args=mini_args,popsize=15,tol=0.01)
                    best_hyperparams = solver.x
                    best_rmse = solver.fun
                    if verbose_level>=1: f_out.write("Best hyperparameters: %f \n" %best_hyperparams)
                    if verbose_level>=1: f_out.write("Best rmse: %f \n"  %best_rmse)
                    if verbose_level>=1: f_out.flush()
                    error_metric_result = best_rmse
            error_metric_list.append(error_metric_result)
            result = error_metric_list
        # Step 2B) Perform t2 exploration
        else:
            # Step 2.B.1) Perform t2 exploration with random biased explorer
            X2a,y2a,unique_t2a = explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,0,unique_t1,t2_time,X1,y1,False)
            # Step 2.B.2) Perform t2 exploration with ML explorer
            X2b,y2b,unique_t2b = explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,0,unique_t1,t2_time,X1,y1,True)
            # select coordinates where minimum is predicted with ML
            x_real=[]
            ## hard coded for 3D ##
            #k_in_grid_2 = [i for i, x in enumerate(dim_list[0]) if x == X2b[np.where(y2b == np.min(y2b))][0][0]]
            #k_in_grid_3 = [i for i, x in enumerate(dim_list[1]) if x == X2b[np.where(y2b == np.min(y2b))][0][1]]
            #k_in_grid_4 = [i for i, x in enumerate(dim_list[2]) if x == X2b[np.where(y2b == np.min(y2b))][0][2]]
            #for i in k_in_grid_2:
                #if i in k_in_grid_3 and i in k_in_grid_4:
                    ##k_in_grid=i
                    #if verbose_level>=1: f_out.write("former new value of k_in_grid is: %s\n" %(str(i)))
                    #if verbose_level>=1: f_out.flush()
            ########################
            ##   For any param   ##
            new_k_in_grid=[[] for j in range(param)]
            for j in range(param):
                k_list = [i for i, x in enumerate(dim_list[j]) if x == X2b[np.where(y2b == np.min(y2b))][0][j]]
                new_k_in_grid[j].append(k_list)
            for i in range(len(k_list)):
                counter_k=0
                for j in range(1,param):
                    if new_k_in_grid[0][0][i] in new_k_in_grid[j][0]:
                        counter_k=counter_k+1
                        if counter_k==param-1:
                            #if verbose_level>=1: f_out.write("new_k_in_grid: %i\n" %(new_k_in_grid[0][0][i]))
                            k_in_grid=new_k_in_grid[0][0][i]
            if verbose_level>=2: f_out.write("new value of k_in_grid is: %i\n" %(k_in_grid))
            if verbose_level>=2: f_out.flush()
            #######################
            # calculate real value of predicted coordinates
            for j in range(param):
                x_real.append(dim_list[j][k_in_grid])
            y_real=G_list[k_in_grid]
            # Calculate ML prediction relative error and MLgain
            error_ML = abs(( min(y2b) - y_real) / y_real)
            MLgain_pred = ( min(y2a) - min(y2b))/abs(min(y2a))
            MLgain_real = ( min(y2a) - y_real)/abs(min(y2a))
            # Print t2 exploration results
            if verbose_level>=1: f_out.write("################ \n")
            if verbose_level>=1: f_out.write("## Initial random exploration: \n")
            if verbose_level>=1: f_out.write("t0 = %i \n" %(t0_time))
            if verbose_level>=1: f_out.write("################ \n")
            if verbose_level>=1: f_out.write("################ \n")
            if verbose_level>=1: f_out.write("## t1 exploration: \n")
            if verbose_level>=1: f_out.write("t1 = %i \n" %(t1_time))
            if verbose_level>=1: f_out.write("Last value: X1 index (unique timestep): %i\n" %(len(y1)-1))
            if verbose_level>=1: f_out.write("Last value: X1: %s, y: %s\n" %(str(X1[-1]),str(y1[-1])))
            if verbose_level>=1: f_out.write("Minimum value: X1 index (unique timestep): %s\n" %(str(np.where(y1 == np.min(y1))[0][0])))
            if verbose_level>=1: f_out.write("Minimum value: X1: %s, y: %s\n" %(str(X1[np.where(y1 == np.min(y1))][0]),str(min(y1))))
            if verbose_level>=1: f_out.write("################ \n")
            if verbose_level>=1: f_out.write("################ \n")
            if verbose_level>=1: f_out.write("## t2 standard exploration: \n")
            if verbose_level>=1: f_out.write("t2 = %i \n" %(t2_time))
            if verbose_level>=1: f_out.write("Last value: X2a index (unique timestep): %i\n" %(len(y2a)-1))
            if verbose_level>=1: f_out.write("Last value: X2a: %s, y2a: %s\n" %(str(X2a[-1]),str(y2a[-1])))
            if verbose_level>=1: f_out.write("Minimum value: X2a index (unique timestep): %s\n" %(str(np.where(y2a == np.min(y2a))[0][0])))
            if verbose_level>=1: f_out.write("Minimum value: X2a: %s, y2a: %s\n" %(str(X2a[np.where(y2a == np.min(y2a))][0]),str(min(y2a))))
            if verbose_level>=1: f_out.write("################ \n")
            if verbose_level>=1: f_out.write("################ \n")
            if verbose_level>=1: f_out.write("## t2 ML exploration: \n")
            if verbose_level>=1: f_out.write("t2 = %i \n" %(t2_time))
            if verbose_level>=1: f_out.write("Last value: X2b index (unique timestep): %i\n" %(len(y2b)-1))
            if verbose_level>=1: f_out.write("Last value: X2b: %s, y2b: %s\n" %(str(X2b[-1]),str(y2b[-1])))
            if verbose_level>=1: f_out.write("Minimum value: X2b index (unique timestep): %s\n" %(str(np.where(y2b == np.min(y2b))[0][0])))
            if verbose_level>=1: f_out.write("Minimum predicted value: X2b: %s, y2b: %s\n" %(str(X2b[np.where(y2b == np.min(y2b))][0]),str(min(y2b))))
            if verbose_level>=1: f_out.write("Minimum real value: X2b: %s, y2b: %s\n" %(str(x_real),str(y_real)))

            if verbose_level>=1: f_out.write("################ \n")
            if verbose_level>=1: f_out.write("################ \n")
            if verbose_level>=1: f_out.write("## ML benefits: \n")
            if verbose_level>=1: f_out.write("Predicted relative MLgain: %.3f%s \n" %(MLgain_pred*100, "%"))
            if verbose_level>=1: f_out.write("Real relative MLgain: %.3f%s \n" %(MLgain_real*100, "%"))
            if verbose_level>=1: f_out.write("ML prediction relative error: %.3f%s \n" %(error_ML*100, "%"))
            if verbose_level>=1: f_out.write("################ \n")

            ML_benefits=[]
            ML_benefits.append(MLgain_pred)
            ML_benefits.append(MLgain_real)
            ML_benefits.append(error_ML)
            #result = MLgain_pred
            if verbose_level>=1: f_out.write("For each Nwalker: %s\n" %(str(ML_benefits)))
            if verbose_level>=1: f_out.flush()
        ML_benefits_list.append(ML_benefits)
        result=ML_benefits_list
        if verbose_level>=1: f_out.write("I am returning this value: %s\n" %(str(result)))
    if verbose_level>=1: f_out.close()
    return result

def generate_grid(iseed,l,f_out):
    Amplitude      = []
    center_N       = [[] for i in range(N)]
    width_N        = [[] for i in range(N)]
    x_list         = []
    dim_list       = [[] for i in range(param)]
    G_list         = []
    if verbose_level>=1: f_out.write('## Start: "generate_grid" function \n')
    if verbose_level>=1: f_out.write("########################### \n")
    if verbose_level>=1: f_out.write("##### Landscape', %i '##### \n" % (l))
    if verbose_level>=1: f_out.write("########################### \n")
    if verbose_level>=1: f_out.write("%s %i %s %6.2f \n" % ('Generated with seed:', iseed, ', and grid_max:', grid_max))
    if verbose_level>=1: f_out.flush()
    # ASSIGN GAUSSIAN VALUES #
    for i in range(N):
        iseed=iseed+1
        random.seed(iseed)
        Amplitude.append(random.uniform(Amplitude_min,Amplitude_max))
        iseed=iseed+1
        random.seed(iseed)
        am_i_negative=random.randint(0,1)
        if am_i_negative==0: Amplitude[i]=-Amplitude[i]
        for dim in range(param):
            iseed=iseed+1
            random.seed(iseed)
            center_N[i].append(random.uniform(center_min,center_max))
            iseed=iseed+1
            random.seed(iseed)
            width_N[i].append(random.uniform(width_min,width_max))
    if param==1:
        if verbose_level>=2: f_out.write("%4s %14s %11s %13s \n" % ("N","Amplitude","Center","Width"))
        for i in range(len(Amplitude)):
            if verbose_level>=2: f_out.write("%4i %2s %10.6f %2s %10.6f %2s %10.6f \n" % (i, "", Amplitude[i],"",center_N[i][0],"",width_N[i][0]))
    elif param==2:
        if verbose_level>=2: f_out.write("%4s %14s %22s %35s \n" % ("N","Amplitude","Center","Width"))
        for i in range(len(Amplitude)):
            if verbose_level>=2: f_out.write("%4i %2s %10.6f %2s %10.6f %10.6f %2s %10.6f %10.6f \n" % (i, "", Amplitude[i],"",center_N[i][0],center_N[i][1],"",width_N[i][0],width_N[i][1]))
    elif param==3:
        if verbose_level>=2: f_out.write("%4s %14s %22s %34s \n" % ("N","Amplitude","Center","Width"))
        for i in range(len(Amplitude)):
            if verbose_level>=2: f_out.write("%4i %2s %10.6f %2s %10.6f %10.6f %10.6f %2s %10.6f %10.6f %10.6f \n" % (i, "", Amplitude[i],"",center_N[i][0],center_N[i][1],center_N[i][2],"",width_N[i][0],width_N[i][1],width_N[i][2]))
    elif param==4:
        if verbose_level>=2: f_out.write("%4s %14s %28s %45s \n" % ("N","Amplitude","Center","Width"))
        for i in range(len(Amplitude)):
            if verbose_level>=2: f_out.write("%4i %2s %10.6f %2s %10.6f %10.6f %10.6f %10.6f %2s %10.6f %10.6f %10.6f %10.6f \n" % (i, "", Amplitude[i],"",center_N[i][0],center_N[i][1],center_N[i][2],center_N[i][3],"",width_N[i][0],width_N[i][1],width_N[i][2],width_N[i][3]))
    elif param==5:
        if verbose_level>=2: f_out.write("%4s %14s %33s %57s \n" % ("N","Amplitude","Center","Width"))
        for i in range(len(Amplitude)):
            if verbose_level>=2: f_out.write("%4i %2s %10.6f %2s %10.6f %10.6f %10.6f %10.6f %10.6f %2s %10.6f %10.6f %10.6f %10.6f %10.6f \n" % (i, "", Amplitude[i],"",center_N[i][0],center_N[i][1],center_N[i][2],center_N[i][3],center_N[i][4],"",width_N[i][0],width_N[i][1],width_N[i][2],width_N[i][3],width_N[i][4]))
    if verbose_level>=2: f_out.flush()

    # CALCULATE G GRID #
    counter=0
    for j in range(param):
        x_list.append(grid_min)
    if param==1:
        if verbose_level>=2: f_out.write("%8s %8s %11s \n" % ("i","x","G"))
        x_list[0]=grid_min
        while x_list[0] < grid_max+grid_Delta/2.0:
            G=0.0
            dim_list[0].append(round(x_list[0],6))
            for i in range(N):
                gauss=0.0
                for dim in range(param):
                    gauss=gauss+((dim_list[dim][counter]-center_N[i][dim])**2/(2.0*width_N[i][dim]**2))
                G = G + Amplitude[i] * math.exp(-gauss)
            G_list.append(G)
            if verbose_level>=2: f_out.write("%8i %2s %6.2f %2s %10.6f \n" % (counter,"",dim_list[0][counter],"",G_list[counter]))
            counter=counter+1
            x_list[0]=x_list[0]+grid_Delta
    elif param==2:
        if verbose_level>=2: f_out.write("%8s %11s %15s \n" % ("i","x","G"))
        x_list[0]=grid_min
        while x_list[0] < grid_max+grid_Delta/2.0:
            x_list[1]=grid_min
            while x_list[1] < grid_max+grid_Delta/2.0:
                G=0.0
                dim_list[0].append(round(x_list[0],6))
                dim_list[1].append(round(x_list[1],6))
                for i in range(N):
                    gauss=0.0
                    for dim in range(param):
                        gauss=gauss+((dim_list[dim][counter]-center_N[i][dim])**2/(2.0*width_N[i][dim]**2))
                    G = G + Amplitude[i] * math.exp(-gauss)
                G_list.append(G)
                if verbose_level>=2: f_out.write("%8i %2s %6.2f %6.2f %2s %10.6f \n" % (counter,"",dim_list[0][counter],dim_list[1][counter],"",G_list[counter]))
                x_list[1]=x_list[1]+grid_Delta
                counter=counter+1
            x_list[0]=x_list[0]+grid_Delta
    elif param==3:
        if verbose_level>=2: f_out.write("%8s %14s %19s \n" % ("i","x","G"))
        x_list[0]=grid_min
        while x_list[0] < grid_max+grid_Delta/2.0:
            x_list[1]=grid_min
            while x_list[1] < grid_max+grid_Delta/2.0:
                x_list[2]=grid_min
                while x_list[2] < grid_max+grid_Delta/2.0:
                    G=0.0
                    dim_list[0].append(round(x_list[0],6))
                    dim_list[1].append(round(x_list[1],6))
                    dim_list[2].append(round(x_list[2],6))
                    for i in range(N):
                        gauss=0.0
                        for dim in range(param):
                            gauss=gauss+((dim_list[dim][counter]-center_N[i][dim])**2/(2.0*width_N[i][dim]**2))
                        G = G + Amplitude[i] * math.exp(-gauss)
                    G_list.append(G)
                    if verbose_level>=2: f_out.write("%8i %2s %6.2f %6.2f %6.2f %2s %10.6f \n" % (counter,"",dim_list[0][counter],dim_list[1][counter],dim_list[2][counter],"",G_list[counter]))
                    x_list[2]=x_list[2]+grid_Delta
                    counter=counter+1
                x_list[1]=x_list[1]+grid_Delta
            x_list[0]=x_list[0]+grid_Delta
    elif param==4:
        if verbose_level>=2: f_out.write("%8s %18s %22s \n" % ("i","x","G"))
        x_list[0]=grid_min
        while x_list[0] <= grid_max+grid_Delta/2.0:
            x_list[1]=grid_min
            while x_list[1] <= grid_max+grid_Delta/2.0:
                x_list[2]=grid_min
                while x_list[2] <= grid_max+grid_Delta/2.0:
                    x_list[3]=grid_min
                    while x_list[3] <= grid_max+grid_Delta/2.0:
                        G=0.0
                        dim_list[0].append(round(x_list[0],6))
                        dim_list[1].append(round(x_list[1],6))
                        dim_list[2].append(round(x_list[2],6))
                        dim_list[3].append(round(x_list[3],6))
                        for i in range(N):
                            gauss=0.0
                            for dim in range(param):
                                gauss=gauss+((dim_list[dim][counter]-center_N[i][dim])**2/(2.0*width_N[i][dim]**2))
                            G = G + Amplitude[i] * math.exp(-gauss)
                        G_list.append(G)
                        if verbose_level>=2: f_out.write("%8i %2s %6.2f %6.2f %6.2f %6.2f %2s %10.6f \n" % (counter,"",dim_list[0][counter],dim_list[1][counter],dim_list[2][counter],dim_list[3][counter],"",G_list[counter]))
                        x_list[3]=x_list[3]+grid_Delta
                        counter=counter+1
                    x_list[2]=x_list[2]+grid_Delta
                x_list[1]=x_list[1]+grid_Delta
            x_list[0]=x_list[0]+grid_Delta
    elif param==5:
        if verbose_level>=2: f_out.write("%8s %22s %25s \n" % ("i","x","G"))
        x_list[0]=grid_min
        while x_list[0] <= grid_max+grid_Delta/2.0:
            x_list[1]=grid_min
            while x_list[1] <= grid_max+grid_Delta/2.0:
                x_list[2]=grid_min
                while x_list[2] <= grid_max+grid_Delta/2.0:
                    x_list[3]=grid_min
                    while x_list[3] <= grid_max+grid_Delta/2.0:
                        x_list[4]=grid_min
                        while x_list[4] <= grid_max+grid_Delta/2.0:
                            G=0.0
                            dim_list[0].append(round(x_list[0],6))
                            dim_list[1].append(round(x_list[1],6))
                            dim_list[2].append(round(x_list[2],6))
                            dim_list[3].append(round(x_list[3],6))
                            dim_list[4].append(round(x_list[4],6))
                            for i in range(N):
                                gauss=0.0
                                for dim in range(param):
                                    gauss=gauss+((dim_list[dim][counter]-center_N[i][dim])**2/(2.0*width_N[i][dim]**2))
                                G = G + Amplitude[i] * math.exp(-gauss)
                            G_list.append(G)
                            if verbose_level>=2: f_out.write("%8i %2s %6.2f %6.2f %6.2f %6.2f %6.2f %2s %10.6f \n" % (counter,"",dim_list[0][counter],dim_list[1][counter],dim_list[2][counter],dim_list[3][counter],dim_list[4][counter],"",G_list[counter]))
                            x_list[4]=x_list[4]+grid_Delta
                            counter=counter+1
                        x_list[3]=x_list[3]+grid_Delta
                    x_list[2]=x_list[2]+grid_Delta
                x_list[1]=x_list[1]+grid_Delta
            x_list[0]=x_list[0]+grid_Delta
    if verbose_level>=1: f_out.flush()

    Ngrid=int((grid_max/grid_Delta+1)**param)   # calculate number of grid points
    max_G=max(G_list)
    min_G=min(G_list)
    if verbose_level>=1:
        max_G_index=int(np.where(G_list == np.max(G_list))[0])
        min_G_index=int(np.where(G_list == np.min(G_list))[0])
        f_out.write("Number of grid points: %i \n" %Ngrid)
        if param==1:
            f_out.write("Maximum value of grid: %f %f  \n" %(dim_list[0][max_G_index], max_G))
            f_out.write("Minimum value of grid: %f %f  \n" %(dim_list[0][min_G_index], min_G))
        if param==2:
            f_out.write("Maximum value of grid: %f %f %f \n" %(dim_list[0][max_G_index], dim_list[1][max_G_index], max_G))
            f_out.write("Minimum value of grid: %f %f %f \n" %(dim_list[0][min_G_index], dim_list[1][min_G_index], min_G))
        if param==3:
            f_out.write("Maximum value of grid: %f %f %f %f \n" %(dim_list[0][max_G_index], dim_list[1][max_G_index], dim_list[2][max_G_index], max_G))
            f_out.write("Minimum value of grid: %f %f %f %f \n" %(dim_list[0][min_G_index], dim_list[1][min_G_index], dim_list[2][min_G_index], min_G))
        if param==4:
            f_out.write("Maximum value of grid: %f %f %f %f %f \n" %(dim_list[0][max_G_index], dim_list[1][max_G_index], dim_list[2][max_G_index], dim_list[3][max_G_index], max_G))
            f_out.write("Minimum value of grid: %f %f %f %f %f \n" %(dim_list[0][min_G_index], dim_list[1][min_G_index], dim_list[2][min_G_index], dim_list[3][min_G_index], min_G))
        if param==5:
            f_out.write("Maximum value of grid: %f %f %f %f %f %f \n" %(dim_list[0][max_G_index], dim_list[1][max_G_index], dim_list[2][max_G_index], dim_list[3][max_G_index], dim_list[4][max_G_index], max_G))
            f_out.write("Minimum value of grid: %f %f %f %f %f %f \n" %(dim_list[0][min_G_index], dim_list[1][min_G_index], dim_list[2][min_G_index], dim_list[3][min_G_index], dim_list[4][min_G_index], min_G))
        f_out.flush()
    return dim_list, G_list, Ngrid, max_G

def check_input_values():
    if type(is_dask) != bool:
        print ('INPUT ERROR: is_dask should be boolean, but is:', is_dask)
        sys.exit()
    if Nwalkers != len(adven):
        print ('INPUT ERROR: Nwalkers is %i, but adven has %i elements:' %(Nwalkers, len(adven)))
        sys.exit()
    if initial_sampling not in allowed_initial_sampling:
        print ('INPUT ERROR: initial_sampling need to be in',allowed_initial_sampling, ', but is:', initial_sampling)
        sys.exit()
    if ML not in allowed_ML:
        print ('INPUT ERROR: ML needs to be in',allowed_ML, ', but is:', ML)
        sys.exit()
    if error_metric not in allowed_error_metric:
        print ('INPUT ERROR: error_metric needs to be in',allowed_error_metric, ', but is:', error_metric)
        sys.exit()
    if CV not in allowed_CV:
        print ('INPUT ERROR: ML needs to be in',allowed_CV, ', but is:', CV)
        sys.exit()
    if verbose_level not in allowed_verbosity_level:
        print ('INPUT ERROR: verbosity_level needs to be in',allowed_verbosity_level, ', but is:', verbosity_level)
    if t2_ML not in allowed_t2_ML:
        print ('INPUT ERROR: ML needs to be in',allowed_t2_ML, ', but is:', t2_ML)
        sys.exit()
        sys.exit()
    print("\n",flush=True)
    print('### START PRINT INPUT ###',flush=True)
    print('# Parallel computing #',flush=True)
    print('is_dask:',is_dask,flush=True)
    print('NCPU',NCPU,flush=True)
    print('# Verbose #',flush=True)
    print('verbose_level:',verbose_level,flush=True)
    print('allowed_verbosity_level:',allowed_verbosity_level,flush=True)
    print('log_name',log_name,flush=True)
    print('# Landscape parameters #',flush=True)
    print('Nspf:',Nspf,flush=True)
    print('S:',S,flush=True)
    print('iseed:',iseed,flush=True)
    print('param:',param,flush=True)
    print('center_min:',center_min,flush=True)
    print('center_max:',center_max,flush=True)
    print('Grid parameters',flush=True)
    print('grid_min:',grid_min,flush=True)
    print('grid_max:',grid_max,flush=True)
    print('grid_Delta:',grid_Delta,flush=True)
    print('# Exploration parameters #',flush=True)
    print('Nwalkers:',Nwalkers,flush=True)
    print('adven:',adven,flush=True)
    print('t1_time:',t1_time,flush=True)
    print('d_threshold:',d_threshold,flush=True)
    print('t0_time:',t0_time,flush=True)
    print('initial_sampling:',initial_sampling,flush=True)
    print('allowed_initial_sampling:',allowed_initial_sampling,flush=True)
    print('# ML algorithm #',flush=True)
    print('allowed_ML:',allowed_ML,flush=True)
    print('allowed_error_metric:',allowed_error_metric,flush=True)
    print('allowed_CV:',allowed_CV,flush=True)
    print('ML:',ML,flush=True)
    print('error_metric:',error_metric,flush=True)
    print('CV:',CV,flush=True)
    print('k_fold:',k_fold,flush=True)
    print('test_last_percentage:',test_last_percentage,flush=True)

    print('t2_time:',t2_time,flush=True)
    print('t2_ML:',t2_ML,flush=True)
    print('allowed_t2_ML:',allowed_t2_ML,flush=True)

    if ML=='kNN':
        print('n_neighbor:',n_neighbor,flush=True)
        print('weights:',weights,flush=True)
    if ML=='GBR':
        print('GBR_criterion:',GBR_criterion,flush=True)
        print('GBR_n_estimators:',GBR_n_estimators,flush=True)
        print('GBR_learning_rate:',GBR_learning_rate,flush=True)
        print('GBR_max_depth:',GBR_max_depth,flush=True)
        print('GBR_min_samples_split:',GBR_min_samples_split,flush=True)
        print('GBR_min_samples_leaf:',GBR_min_samples_leaf,flush=True)
    if ML=='GPR':
        print('A_RBF:',A_RBF,flush=True)
        print('A_noise:',A_noise,flush=True)
        print('GPR_alpha:',GPR_alpha,flush=True)
        print('kernel_length_scale:',kernel_length_scale,flush=True)
        print('kernel_noise_level:',kernel_noise_level,flush=True)
    if ML=='KRR':
        print('KRR_alpha:',KRR_alpha,flush=True)
        print('KRR_kernel:',KRR_kernel,flush=True)
        print('KRR_gamma:',KRR_gamma,flush=True)
        print('optimize_gamma:',optimize_gamma,flush=True)
        print('KRR_gamma_lim:',KRR_gamma_lim,flush=True)
    print('Calculated width_min:',width_min,flush=True)
    print('Calculated width_max:',width_max,flush=True)
    print('Calculated Amplitude_min:',Amplitude_min,flush=True)
    print('Calculated Amplitude_max:',Amplitude_max,flush=True)
    print('Calculated N:',N,flush=True)
    print('#### END PRINT INPUT ####',flush=True)
    print("\n",flush=True)

def explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,t0,t1,t2,Xi,yi,ML_explore):
    walker_x       = []
    path_x         = [[] for i in range(param)]
    path_G         = []
    list_t         = []
    prob           = []
    neighbor_walker= [[] for i in range(param)]
    neighbor_G     = []
    # print header
    if verbose_level>=1: f_out.write('## Start: "explore_landscape" function \n')
    if verbose_level>=1: f_out.write("############# \n")
    if verbose_level>=1: f_out.write("Start explorer %i \n" % (w))
    if verbose_level>=1: f_out.write("Adventurousness: %f \n" % (adven[w]))
    if verbose_level>=1: f_out.write("############# \n")
    Nx=((grid_max-grid_min)/grid_Delta)+1
    if verbose_level>=1: f_out.write("Number of points per dimension: %i \n" %Nx)
    if verbose_level>=1: f_out.write("Testing w: %i, iseed: %i \n" % (w,iseed))
    if verbose_level>=1: f_out.flush()
    # START UNBIASED RANDOM EXPLORATION #
    # Perform t0 exploration
    for t in range(t0):
        for i in range(param):
            if initial_sampling=='different': iseed=iseed+w+l+i+t
            if initial_sampling=='same':      iseed=iseed+1
            random.seed(iseed)
            num=int(random.randint(0,Ngrid-1))
            if t==0:
                walker_x.append(dim_list[i][num])
            else:
                walker_x[i]=dim_list[i][num]
        for i in range(param):
            path_x[i].append(walker_x[i])
        num_in_grid=0.0
        for i in range(param):
            num_in_grid=num_in_grid + walker_x[param-1-i]*(Nx**i)*(Nx-1)
        num_in_grid=int(round(num_in_grid))
        path_G.append(G_list[num_in_grid])
        list_t.append(t)
        if param==1 and verbose_level>=1: f_out.write("timestep %4i %2s %6.2f %2s %10.6f \n" % (t,"",walker_x[0],"",G_list[num_in_grid]))
        if param==2 and verbose_level>=1: f_out.write("timestep %4i %2s %6.2f %6.2f %2s %10.6f \n" % (t,"",walker_x[0],walker_x[1],"",G_list[num_in_grid]))
        if param==3 and verbose_level>=1: f_out.write("timestep %4i %2s %6.2f %6.2f %6.2f %2s %10.6f \n" % (t,"",walker_x[0],walker_x[1],walker_x[2],"",G_list[num_in_grid]))
        if param==4 and verbose_level>=1: f_out.write("timestep %4i %2s %6.2f %6.2f %6.2f %6.2f %2s %10.6f \n" % (t,"",walker_x[0],walker_x[1],walker_x[2],walker_x[3],"",G_list[num_in_grid]))
        if param==5 and verbose_level>=1: f_out.write("timestep %4i %2s %6.2f %6.2f %6.2f %6.2f %6.2f %2s %10.6f\n" % (t,"",walker_x[0],walker_x[1],walker_x[2],walker_x[3],walker_x[4],"",G_list[num_in_grid]))
        if verbose_level>=1: f_out.flush()
    x_param=[[] for j in range(param)]
    y=[]
    # Set values for t1 exploration
    if t2==0:
        t_ini=t0
        t_fin=t1+t0
        for i in range(param): # set walker_x to last path_x
            walker_x[i]=path_x[i][-1]
    # Set values for t2 exploration
    elif t0==0:
        t_ini=0
        t_fin=t2
        for t in range(t1): # copy Xi and yi to path_x and path_G
            for i in range(param):
                path_x[i].append(Xi[t][i])
            path_G.append(yi[t])
        for i in range(param): # set walker_x to last path_x
            walker_x.append(Xi[-1][i])
    # CONTINUE BIASED RANDOM EXPLORATION #
    if ML_explore == False:
        if verbose_level>=1: f_out.write("## Start random biased exploration\n" %())
        if verbose_level>=1: f_out.flush()
        for t in range(t_ini,t_fin):
            del prob[:]
            del neighbor_walker[:][:]
            del neighbor_G[:]
            prob_sum=0.0
            # Calculate number of special points
            minimum_path_G = []
            minimum_path_E = []
            minimum_path_x = [[] for i in range(param)]
            for i in np.argsort(path_G):
                for j in range(param):
                    minimum_path_x[j].append(path_x[j][i])
                minimum_path_G.append(path_G[i])
            minimum_path_E=[abs(E-max_G) for E in minimum_path_G]
            special_points=int(round(adven[w]/100*len(minimum_path_G)))
            if special_points<1: special_points=1
            draw=random.choice(range(special_points))
            draw_in_grid=G_list.index(minimum_path_G[draw])
            draw_in_grid_list=[i for i, e in enumerate(G_list) if e == minimum_path_G[draw] ]
            if verbose_level>=2: f_out.write("Special points: %i \n" % (special_points))
            for i in range(special_points):
                if param==1 and verbose_level>=2: f_out.write("%i %6.2f %10.6f \n" % (i,minimum_path_x[0][i],minimum_path_G[i]))
                if param==2 and verbose_level>=2: f_out.write("%i %6.2f %6.2f %10.6f \n" % (i,minimum_path_x[0][i],minimum_path_x[1][i],minimum_path_G[i]))
                if param==3 and verbose_level>=2: f_out.write("%i %6.2f %6.2f %6.2f %10.6f \n" % (i,minimum_path_x[0][i],minimum_path_x[1][i],minimum_path_x[2][i],minimum_path_G[i]))
                if param==4 and verbose_level>=2: f_out.write("%i %6.2f %6.2f %6.2f %6.2f %10.6f \n" % (i,minimum_path_x[0][i],minimum_path_x[1][i],minimum_path_x[2][i],minimum_path_x[3][i],minimum_path_G[i]))
                if param==5 and verbose_level>=2: f_out.write("%i %6.2f %6.2f %6.2f %6.2f %6.2f %10.6f \n" % (i,minimum_path_x[0][i],minimum_path_x[1][i],minimum_path_x[2][i],minimum_path_x[3][i],minimum_path_x[4][i],minimum_path_G[i]))
            for i in range(param):
                if minimum_path_x[i][draw] != dim_list[i][draw_in_grid]:
                    print("STOP - ERROR: minimum_path not equal to dum_list (maybe more than 1 point with that value in grid)")
                    print("Selected point draw:", minimum_path_x[:][draw],minimum_path_G[draw])
                    print("Selected point draw in grid:", dim_list[:][draw_in_grid],G_list[draw_in_grid])
                    sys.exit()
            if verbose_level>=1: f_out.flush()
            P=int(round(d_threshold/grid_Delta + 1))
            if verbose_level>=2: f_out.write("Consider nearby points: \n")
            if verbose_level>=1: f_out.flush()
            counter3=0
            for i in range((P*2+1)**param):
                    prob.append(0.0)
                    neighbor_G.append(0.0)
                    for j in range(param):
                        neighbor_walker[j].append(0.0)
            if param==1:
                if verbose_level>=2: f_out.write("%6s %5s %11s %11s %13s \n" % ("i","x","G","Prob","distance"))
                for i1 in range(-P+1,P):
                    try:
                        index0=int(round(draw_in_grid-i1))
                        indexG=int(round(draw_in_grid-i1))
                        d_ij=round((math.sqrt((minimum_path_x[0][draw]-dim_list[0][index0])**2)),6)
                        if d_ij < d_threshold and d_ij > 0.0:
                            neighbor_walker[0][counter3]=dim_list[0][index0]
                            neighbor_G[counter3]=G_list[indexG]
                            prob[counter3]=1.0
                            prob_sum=prob_sum+prob[counter3]
                        if verbose_level>=2: f_out.write("%6i %6.2f %2s %10.6f %2s %5.1f %2s %10.6f \n" % (counter3,dim_list[0][index0],"",G_list[indexG],"",prob[counter3],"",d_ij))
                    except:
                        pass
                    counter3=counter3+1
            elif param==2:
                if verbose_level>=2: f_out.write("%6s %8s %15s %11s %13s \n" % ("i","x","G","Prob","distance"))
                for i2 in range(-P+1,P):
                    for i1 in range(-P+1,P):
                        try:
                            index0=int(round(draw_in_grid-Nx*i2))
                            index1=int(round(draw_in_grid-i1))
                            indexG=int(round(draw_in_grid-Nx*i2-i1))
                            d_ij=round((math.sqrt((minimum_path_x[0][draw]-dim_list[0][index0])**2+(minimum_path_x[1][draw]-dim_list[1][index1])**2)),6)
                            if d_ij < d_threshold and d_ij > 0.0:
                                neighbor_walker[0][counter3]=dim_list[0][index0]
                                neighbor_walker[1][counter3]=dim_list[1][index1]
                                neighbor_G[counter3]=G_list[indexG]
                                prob[counter3]=1.0
                                prob_sum=prob_sum+prob[counter3]
                            if verbose_level>=2: f_out.write("%6i %6.2f %6.2f %2s %10.6f %2s %5.1f %2s %10.6f \n" % (counter3,dim_list[0][index0],dim_list[1][index1],"",G_list[indexG],"",prob[counter3],"",d_ij))
                        except:
                            pass
                        counter3=counter3+1
            elif param==3:
                if verbose_level>=2: f_out.write("%6s %11s %19s %12s %12s \n" % ("i","x","G","Prob","distance"))
                for i3 in range(-P+1,P):
                    for i2 in range(-P+1,P):
                        for i1 in range(-P+1,P):
                            try:
                                index0=int(round(draw_in_grid-Nx*Nx*i3))
                                index1=int(round(draw_in_grid-Nx*i2))
                                index2=int(round(draw_in_grid-i1))
                                indexG=int(round(draw_in_grid-Nx*Nx*i3-Nx*i2-i1))
                                d_ij=round((math.sqrt((minimum_path_x[0][draw]-dim_list[0][index0])**2+(minimum_path_x[1][draw]-dim_list[1][index1])**2+(minimum_path_x[2][draw]-dim_list[2][index2])**2)),6)
                                if d_ij < d_threshold and d_ij > 0.0:
                                    neighbor_walker[0][counter3]=dim_list[0][index0]
                                    neighbor_walker[1][counter3]=dim_list[1][index1]
                                    neighbor_walker[2][counter3]=dim_list[2][index2]
                                    neighbor_G[counter3]=G_list[indexG]
                                    prob[counter3]=1.0
                                    prob_sum=prob_sum+prob[counter3]
                                if verbose_level>=2: f_out.write("%6i %6.2f %6.2f %6.2f %2s %10.6f %2s %5.1f %2s %10.6f \n" % (counter3,dim_list[0][index0],dim_list[1][index1],dim_list[2][index2],"",G_list[indexG],"",prob[counter3],"",d_ij))
                            except:
                                pass
                            counter3=counter3+1
            elif param==4:
                if verbose_level>=2: f_out.write("%6s %15s %22s %12s %12s \n" % ("i","x","G","Prob","distance"))
                for i4 in range(-P+1,P):
                    for i3 in range(-P+1,P):
                        for i2 in range(-P+1,P):
                            for i1 in range(-P+1,P):
                                try:
                                    index0=int(round(draw_in_grid-Nx*Nx*Nx*i4))
                                    index1=int(round(draw_in_grid-Nx*Nx*i3))
                                    index2=int(round(draw_in_grid-Nx*i2))
                                    index3=int(round(draw_in_grid-i1))
                                    indexG=int(round(draw_in_grid-Nx*Nx*Nx*i4-Nx*Nx*i3-Nx*i2-i1))
                                    d_ij=round((math.sqrt((minimum_path_x[0][draw]-dim_list[0][index0])**2+(minimum_path_x[1][draw]-dim_list[1][index1])**2+(minimum_path_x[2][draw]-dim_list[2][index2])**2+(minimum_path_x[3][draw]-dim_list[3][index3])**2)),6)
                                    if d_ij < d_threshold and d_ij > 0.0:
                                        neighbor_walker[0][counter3]=dim_list[0][index0]
                                        neighbor_walker[1][counter3]=dim_list[1][index1]
                                        neighbor_walker[2][counter3]=dim_list[2][index2]
                                        neighbor_walker[3][counter3]=dim_list[3][index3]
                                        neighbor_G[counter3]=G_list[indexG]
                                        prob[counter3]=1.0
                                        prob_sum=prob_sum+prob[counter3]
                                    if verbose_level>=2: f_out.write("%6i %6.2f %6.2f %6.2f %6.2f %2s %10.6f %2s %5.1f %2s %10.6f \n" % (counter3,dim_list[0][index0],dim_list[1][index1],dim_list[2][index2],dim_list[3][index3],"",G_list[indexG],"",prob[counter3],"",d_ij))
                                except:
                                    pass
                                counter3=counter3+1
            elif param==5:
                if verbose_level>=2: f_out.write("%6s %19s %25s %12s %12s \n" % ("i","x","G","Prob","distance"))
                for i5 in range(-P+1,P):
                    for i4 in range(-P+1,P):
                        for i3 in range(-P+1,P):
                            for i2 in range(-P+1,P):
                                for i1 in range(-P+1,P):
                                    try:
                                        index0=int(round(draw_in_grid-Nx*Nx*Nx*Nx*i5))
                                        index1=int(round(draw_in_grid-Nx*Nx*Nx*i4))
                                        index2=int(round(draw_in_grid-Nx*Nx*i3))
                                        index3=int(round(draw_in_grid-Nx*i2))
                                        index4=int(round(draw_in_grid-i1))
                                        indexG=int(round(draw_in_grid-Nx*Nx*Nx*Nx*i5-Nx*Nx*Nx*i4-Nx*Nx*i3-Nx*i2-i1))
                                        d_ij=round((math.sqrt((minimum_path_x[0][draw]-dim_list[0][index0])**2+(minimum_path_x[1][draw]-dim_list[1][index1])**2+(minimum_path_x[2][draw]-dim_list[2][index2])**2+(minimum_path_x[3][draw]-dim_list[3][index3])**2+(minimum_path_x[4][draw]-dim_list[4][index4])**2)),6)
                                        if d_ij < d_threshold and d_ij > 0.0:
                                            neighbor_walker[0][counter3]=dim_list[0][index0]
                                            neighbor_walker[1][counter3]=dim_list[1][index1]
                                            neighbor_walker[2][counter3]=dim_list[2][index2]
                                            neighbor_walker[3][counter3]=dim_list[3][index3]
                                            neighbor_walker[4][counter3]=dim_list[4][index4]
                                            neighbor_G[counter3]=G_list[indexG]
                                            prob[counter3]=1.0
                                            prob_sum=prob_sum+prob[counter3]
                                        if verbose_level>=2: f_out.write("%6i %6.2f %6.2f %6.2f %6.2f %6.2f %2s %10.6f %2s %5.1f %2s %10.6f \n" % (counter3,dim_list[0][index0],dim_list[1][index1],dim_list[2][index2],dim_list[3][index3],dim_list[4][index4],"",G_list[indexG],"",prob[counter3],"",d_ij))
                                    except:
                                        pass
                                    counter3=counter3+1
            if verbose_level>=1: f_out.flush()
            if len(range((P*2+1)**param)) != len(prob):
                print("STOP - ERROR: Problem with number of nearby points considered for next step",flush=True)
                sys.exit()
            if verbose_level>=2: f_out.write("Number of points considered: %i \n" % (len(range((P*2+1)**param))))
            if verbose_level>=2: f_out.write("Points within threshold: %f \n" % int(round((prob_sum))))
            if verbose_level>=1: f_out.flush()
    
            for i in range(counter3):
                prob[i]=prob[i]/prob_sum
            draw=int(choice(range((P*2+1)**param),size=1,p=prob))
            for i in range(param):
                walker_x[i]=neighbor_walker[i][draw]
                path_x[i].append(walker_x[i])
            path_G.append(neighbor_G[draw])
            list_t.append(t)
            if verbose_level>=2: f_out.write("We draw neighbor no.: %6i\n" % (draw))
            if verbose_level>=1: f_out.flush()
            if param==1 and verbose_level>=1: f_out.write("timestep %6i %2s %6.2f %2s %10.6f\n" % (t,"",walker_x[0],"",neighbor_G[draw]))
            if param==2 and verbose_level>=1: f_out.write("timestep %6i %2s %6.2f %6.2f %2s %10.6f\n" % (t,"",walker_x[0],walker_x[1],"",neighbor_G[draw]))
            if param==3 and verbose_level>=1: f_out.write("timestep %6i %2s %6.2f %6.2f %6.2f %2s %10.6f\n" % (t,"",walker_x[0],walker_x[1],walker_x[2],"",neighbor_G[draw]))
            if param==4 and verbose_level>=1: f_out.write("timestep %6i %2s %6.2f %6.2f %6.2f %6.2f %2s %10.6f\n" % (t,"",walker_x[0],walker_x[1],walker_x[2],walker_x[3],"",neighbor_G[draw]))
            if param==5 and verbose_level>=1: f_out.write("timestep %6i %2s %6.2f %6.2f %6.2f %6.2f %6.2f %2s %10.6f\n" % (t,"",walker_x[0],walker_x[1],walker_x[2],walker_x[3],walker_x[4],"",neighbor_G[draw]))
            if verbose_level>=1: f_out.flush()
            for i in range(param):
                x_param[i].append(walker_x[i])
            y.append(neighbor_G[draw])
        X,y = create_X_and_y(f_out,x_param,y)
    # CONTINUE ML EXPLORATION #
    if ML_explore == True:
        for t in range(t_ini,t_fin):
            # initialize values
            x_bubble=[[] for j in range(param)]
            y_bubble=[]
            if verbose_level>=1: f_out.write("## Start ML Exploration\n" %())
            if verbose_level>=1: f_out.flush()
            P=int(round(d_threshold/grid_Delta + 1))
            # For each point in Xi
            for k in range(len(path_G)):
                counter3=0
                del prob[:]
                del neighbor_walker[:][:]
                del neighbor_G[:]
                prob_sum=0.0
                ## hard coded for 3D ##
                #k_in_grid_2 = [i for i, x in enumerate(dim_list[0]) if x == path_x[0][k]]
                #k_in_grid_3 = [i for i, x in enumerate(dim_list[1]) if x == path_x[1][k]]
                #k_in_grid_4 = [i for i, x in enumerate(dim_list[2]) if x == path_x[2][k]]
                #if verbose_level>=1: f_out.write("k_in_grid_2: %s, length: %i\n" %(str(k_in_grid_2),len(k_in_grid_2)))
                #if verbose_level>=1: f_out.write("k_in_grid_3: %s, length: %i\n" %(str(k_in_grid_3),len(k_in_grid_3)))
                #if verbose_level>=1: f_out.write("k_in_grid_4: %s, length: %i\n" %(str(k_in_grid_4),len(k_in_grid_4)))
                #for i in k_in_grid_2:
                    #for j in k_in_grid_3:
                        #if i in k_in_grid_4 and i in k_in_grid_3:
                            #k_in_grid=i
                #if verbose_level>=2: f_out.write("value of k_in_grid is: %i\n" %(k_in_grid))
                #######################
                ##   For any param   ##
                new_k_in_grid=[[] for j in range(param)]
                for j in range(param):
                    k_list = [i for i, x in enumerate(dim_list[j]) if x == path_x[j][k]]
                    new_k_in_grid[j].append(k_list)
                for i in range(len(k_list)):
                    counter_k=0
                    for j in range(1,param):
                        if new_k_in_grid[0][0][i] in new_k_in_grid[j][0]:
                            counter_k=counter_k+1
                            if counter_k==param-1:
                                #if verbose_level>=1: f_out.write("new_k_in_grid: %i\n" %(new_k_in_grid[0][0][i]))
                                k_in_grid=new_k_in_grid[0][0][i]
                if verbose_level>=2: f_out.write("value of k_in_grid is: %i\n" %(k_in_grid))
                if verbose_level>=2: f_out.flush()
                #######################
                # initialize neighbor_G and neighbor_walker[j]
                for i in range((P*2+1)**param):
                    prob.append(0.0)
                    neighbor_G.append(0.0)
                    for j in range(param):
                        neighbor_walker[j].append(0.0)
                # Check points within threshold
                if param==2:
                    if verbose_level>=2: f_out.write("Check around point: %f %f %f\n" %(path_x[0][k], path_x[1][k], path_G[k]))
                    if verbose_level>=2: f_out.write("%6s %8s %15s %11s %13s \n" % ("i","x","G","Prob","distance"))
                    for i2 in range(-P+1,P):
                        for i1 in range(-P+1,P):
                            try:
                                index0=int(round(k_in_grid-Nx*i2))
                                index1=int(round(k_in_grid-i1))
                                indexG=int(round(k_in_grid-Nx*i2-i1))
                                d_ij=round((math.sqrt((path_x[0][k]-dim_list[0][index0])**2+(path_x[1][k]-dim_list[1][index1])**2)),6)
                                if d_ij < d_threshold and d_ij > 0.0:
                                    use_this_point=True
                                    # need to specify that this point is different than previous points
                                    for i in range(len(path_G)):
                                        if dim_list[0][index0] == path_x[0][i] and dim_list[1][index1] == path_x[1][i]:
                                            use_this_point=False
                                    if use_this_point==True:
                                        neighbor_walker[0][counter3]=dim_list[0][index0]
                                        neighbor_walker[1][counter3]=dim_list[1][index1]
                                        neighbor_G[counter3]=G_list[indexG]
                                        prob[counter3]=1.0
                                        prob_sum=prob_sum+prob[counter3]
                                    for j in range(param):
                                        x_bubble[j].append(neighbor_walker[j][counter3])
                                    y_bubble.append(neighbor_G[counter3])
                                if verbose_level>=2: f_out.write("%6i %6.2f %6.2f %2s %10.6f %2s %5.1f %2s %10.6f \n" % (counter3,dim_list[0][index0],dim_list[1][index1],"",G_list[indexG],"",prob[counter3],"",d_ij))
                            except:
                                pass
                            counter3=counter3+1
                elif param==3:
                    if verbose_level>=2: f_out.write("Check around point: %f %f %f %f\n" %(path_x[0][k], path_x[1][k], path_x[2][k], path_G[k]))
                    if verbose_level>=2: f_out.write("%6s %11s %19s %12s %12s \n" % ("i","x","G","Prob","distance"))
                    for i3 in range(-P+1,P):
                        for i2 in range(-P+1,P):
                            for i1 in range(-P+1,P):
                                try:
                                    index0=int(round(k_in_grid-Nx*Nx*i3))
                                    index1=int(round(k_in_grid-Nx*i2))
                                    index2=int(round(k_in_grid-i1))
                                    indexG=int(round(k_in_grid-Nx*Nx*i3-Nx*i2-i1))
                                    d_ij=round((math.sqrt((path_x[0][k]-dim_list[0][index0])**2+(path_x[1][k]-dim_list[1][index1])**2+(path_x[2][k]-dim_list[2][index2])**2)),6)
                                    if d_ij < d_threshold and d_ij > 0.0:
                                        use_this_point=True
                                        # need to specify that this point is different than previous points
                                        for i in range(len(path_G)):
                                            if dim_list[0][index0] == path_x[0][i] and dim_list[1][index1] == path_x[1][i] and dim_list[2][index2] == path_x[2][i]:
                                                use_this_point=False
                                        if use_this_point==True:
                                            neighbor_walker[0][counter3]=dim_list[0][index0]
                                            neighbor_walker[1][counter3]=dim_list[1][index1]
                                            neighbor_walker[2][counter3]=dim_list[2][index2]
                                            neighbor_G[counter3]=G_list[indexG]
                                            prob[counter3]=1.0
                                            prob_sum=prob_sum+prob[counter3]
                                        for j in range(param):
                                            x_bubble[j].append(neighbor_walker[j][counter3])
                                        y_bubble.append(neighbor_G[counter3])
                                    if verbose_level>=2: f_out.write("%6i %6.2f %6.2f %6.2f %2s %10.6f %2s %5.1f %2s %10.6f \n" % (counter3,dim_list[0][index0],dim_list[1][index1],dim_list[2][index2],"",G_list[indexG],"",prob[counter3],"",d_ij))
                                except:
                                    pass
                                counter3=counter3+1
            if verbose_level>=2:
                for j in range(param):
                    f_out.write("x_bubble[%i]: %s\n" %(j,str(x_bubble[j])))
            # train ML with path_x and path_G
            # predict ML for x_bubble and y_bubble
            path_x,path_G = create_X_and_y(f_out,path_x,path_G)
            x_bubble,y_bubble = create_X_and_y(f_out,x_bubble,y_bubble)
            if t2_ML=='kNN': min_point=kNN(x_bubble,y_bubble,iseed,l,w,f_out,path_x,path_G,2)
            if t2_ML=='GPR': min_point=GPR(x_bubble,y_bubble,iseed,l,w,f_out,path_x,path_G,2)
            # select x_bubble corresponding to min(y_bubble)
            if verbose_level>=1: f_out.write("## At time %i, minimum predicted point is: %s\n" %(t, str(min_point)))
            # prepare for next timestep
            path_x=path_x.tolist()
            path_G=path_G.tolist()
            path_x.append(min_point[0:param])
            path_G.append(min_point[param])
            path_x=[list(i) for i in zip(*path_x)]
            # add to final array
            for i in range(param):
                x_param[i].append(min_point[i])
            y.append(min_point[param])
        X,y = create_X_and_y(f_out,x_param,y)
    return X,y,len(y)

# CREATE X AND y #
def create_X_and_y(f_out,x_param,y):
    if param==1:
        zippedList = list(zip(x_param[0],y))
        df=pd.DataFrame(zippedList,columns=['x_param[0]','y'])
        df2=df.drop_duplicates(subset='y', keep="last")
        #df2=df
        df_X=df2[['x_param[0]']]
        df_y=df2['y']
    elif param==2:
        zippedList = list(zip(x_param[0],x_param[1],y))
        df=pd.DataFrame(zippedList,columns=['x_param[0]','x_param[1]','y'])
        df2=df.drop_duplicates(subset='y', keep="last")
        #df2=df
        df_X=df2[['x_param[0]','x_param[1]']]
        df_y=df2['y']
    elif param==3:
        zippedList = list(zip(x_param[0],x_param[1],x_param[2],y))
        df=pd.DataFrame(zippedList,columns=['x_param[0]','x_param[1]','x_param[2]','y'])
        df2=df.drop_duplicates(subset='y', keep="last")
        #df2=df
        df_X=df2[['x_param[0]','x_param[1]','x_param[2]']]
        df_y=df2['y']
    elif param==4:
        zippedList = list(zip(x_param[0],x_param[1],x_param[2],x_param[3],y))
        df=pd.DataFrame(zippedList,columns=['x_param[0]','x_param[1]','x_param[2]','x_param[3]','y'])
        df2=df.drop_duplicates(subset='y', keep="last")
        #df2=df
        df_X=df2[['x_param[0]','x_param[1]','x_param[2]','x_param[3]']]
        df_y=df2['y']
    elif param==5:
        zippedList = list(zip(x_param[0],x_param[1],x_param[2],x_param[3],x_param[4],y))
        df=pd.DataFrame(zippedList,columns=['x_param[0]','x_param[1]','x_param[2]','x_param[3]','x_param[4]','y'])
        df2=df.drop_duplicates(subset='y', keep="last")
        #df2=df
        df_X=df2[['x_param[0]','x_param[1]','x_param[2]','x_param[3]','x_param[3]']]
        df_y=df2['y']
    X=df_X.to_numpy()
    y=df_y.to_numpy()
    if verbose_level>=1: f_out.write("## Unique points: %i\n" % (len(y)))
    if verbose_level>=1: f_out.write("## X: \n")
    if verbose_level>=1: f_out.write("%s \n" % (str(X)))
    if verbose_level>=1: f_out.write("## y: \n")
    if verbose_level>=1: f_out.write("%s \n" % (str(y)))
    if verbose_level>=1: f_out.flush()
    return X,y

# CALCULATE k-NN #
def kNN(X,y,iseed,l,w,f_out,Xtr,ytr,mode):
    iseed=iseed+1
    # Calculate rmse metric
    if mode==1:
        if verbose_level>=1: f_out.write('## Start: "kNN" function \n')
        if verbose_level>=1: f_out.write('-------- \n')
        if verbose_level>=1: f_out.write('Perform k-NN \n')
        if verbose_level>=1: f_out.write('k= %i \n' % (n_neighbor))
        if verbose_level>=1: f_out.write('cross_validation %i - fold \n' % (k_fold))
        if verbose_level>=1: f_out.write('weights %s \n' % (weights))
        if verbose_level>=1: f_out.write('iseed %s \n' % (iseed))
        if verbose_level>=1: f_out.write('-------- \n')
        if verbose_level>=1: f_out.flush()
    
        kf = KFold(n_splits=k_fold,shuffle=True,random_state=iseed)
        n_neighbors = n_neighbor
        average_r=0.0
        average_r_pearson=0.0
        average_rmse=0.0
        if CV=='kf':
            kf = KFold(n_splits=k_fold,shuffle=True,random_state=iseed)
            validation=kf.split(X)
        if CV=='loo':
            loo = LeaveOneOut()
            validation=loo.split(X)
        if CV=='kf' or CV=='loo':
            counter=0
            real_y=[]
            predicted_y=[]
            for train_index, test_index in validation:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
                y_pred = knn.fit(X_train, y_train).predict(X_test)
                for i in range(len(y_test)):
                    #f_out.write("y_test[i] %s \n" %(str(y_test[i])))
                    real_y.append(y_test[i])
                for i in range(len(y_pred)):
                    #f_out.write("y_pred[i] %s \n" %(str(y_pred[i])))
                    predicted_y.append(y_pred[i]) #
                if CV=='kf':
                    r_pearson,_=pearsonr(y_test,y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    if verbose_level>=2: f_out.write('Landscape %i . Adventurousness: %i . k-fold: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],counter,r_pearson,rmse))
                    if verbose_level>=2: f_out.write("%i test points: %s \n" % (len(test_index),str(test_index)))
                    counter=counter+1
                    average_r_pearson=average_r_pearson+r_pearson
                    average_rmse=average_rmse+rmse
            if CV=='kf':
                average_r_pearson=average_r_pearson/k_fold
                average_rmse=average_rmse/k_fold
                if verbose_level>=2: f_out.write('k-fold average r_pearson score: %f \n' % (average_r_pearson))
                if verbose_level>=2: f_out.write('k-fold average rmse score: %f \n' % (average_rmse))
            total_r_pearson,_ = pearsonr(real_y,predicted_y)
            total_mse = mean_squared_error(real_y, predicted_y)
            total_rmse = np.sqrt(total_mse)
        elif CV=='sort':
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_last_percentage,random_state=iseed,shuffle=False)
            knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
            y_pred = knn.fit(X_train, y_train).predict(X_test)
            total_r_pearson,_=pearsonr(y_test,y_pred)
            mse = mean_squared_error(y_test, y_pred)
            total_rmse = np.sqrt(mse)
            if verbose_level>=1: f_out.write("Train with first %i points \n" % (len(X_train)))
            if verbose_level>=1: f_out.write("%s \n" % (str(X_train)))
            if verbose_level>=1: f_out.write("Test with last %i points \n" % (len(X_test)))
            if verbose_level>=1: f_out.write("%s \n" % (str(X_test)))
            if verbose_level>=1: f_out.write('Landscape %i . Adventurousness: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],total_r_pearson,total_rmse))
        if verbose_level>=1: f_out.write('Final r_pearson score: %f \n' % (total_r_pearson))
        if verbose_level>=1: f_out.write('Final rmse score: %f \n' % (total_rmse))
        if error_metric=='rmse': result=total_rmse
        if verbose_level>=1: f_out.flush()
    elif mode==2:
        if verbose_level>=1: f_out.write('## Start: "kNN" function \n')
        if verbose_level>=1: f_out.write('-------- \n')
        if verbose_level>=1: f_out.write('Perform k-NN \n')
        if verbose_level>=1: f_out.write('k= %i \n' % (n_neighbor))
        if verbose_level>=1: f_out.write('weights %s \n' % (weights))
        if verbose_level>=1: f_out.write('iseed %s \n' % (iseed))
        if verbose_level>=1: f_out.write('-------- \n')
        if verbose_level>=1: f_out.flush()
    
        n_neighbors = n_neighbor
        X_train, X_test = Xtr, X
        y_train, y_test = ytr, y
        real_y=[]
        predicted_y=[]
        result = []
        if verbose_level>=2: f_out.write("X_train: %s\n" %(str(X_train)))
        if verbose_level>=2: f_out.write("y_train: %s\n" %(str(y_train)))
        if verbose_level>=2: f_out.write("X_test: %s\n" %(str(X_test)))
        if verbose_level>=2: f_out.write("y_test: %s\n" %(str(y_test)))

        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        #y_pred = knn.fit(X_train, y_train).predict(X_test)
        knn.fit(X_train, y_train)
        y_pred=knn.predict(X_test)
        for i in range(len(y_test)):
            #f_out.write("y_test[i] %s \n" %(str(y_test[i])))
            real_y.append(y_test[i])
        for i in range(len(y_pred)):
            #f_out.write("y_pred[i] %s \n" %(str(y_pred[i])))
            predicted_y.append(y_pred[i]) #
        min_index = predicted_y.index(min(predicted_y))
        if verbose_level>=2: f_out.write("At index %i, predicted minimum value: %f\n" %(min_index, min(predicted_y)))
        if verbose_level>=2: f_out.write("At index %i, 'real' minimum value: %f\n" %(min_index, min(real_y)))
        for j in range(param):
            result.append(X_test[min_index][j])
        result.append(min(predicted_y))
    return result

# CALCULATE GBR #
def GBR(X,y,iseed,l,w,f_out):
    iseed=iseed+1
    if verbose_level>=1: f_out.write('## Start: "GBR" function \n')
    if verbose_level>=1: f_out.write('-------- \n')
    if verbose_level>=1: f_out.write('Perform GBR\n')
    if verbose_level>=1: f_out.write('cross_validation %i - fold\n' % (k_fold))
    if verbose_level>=1: f_out.write('GBR criterion: %s\n' % (GBR_criterion))
    if verbose_level>=1: f_out.write('Number of estimators: %i\n' % (GBR_n_estimators))
    if verbose_level>=1: f_out.write('Learning rate: %f\n' % (GBR_learning_rate))
    if verbose_level>=1: f_out.write('Tree max depth: %i\n' % (GBR_max_depth))
    if verbose_level>=1: f_out.write('Min samples to split: %i\n' % (GBR_min_samples_split))
    if verbose_level>=1: f_out.write('Min samples per leaf: %i\n' % (GBR_min_samples_leaf))
    if verbose_level>=1: f_out.write('--------\n')
    if verbose_level>=1: f_out.flush()

    kf = KFold(n_splits=k_fold,shuffle=True,random_state=iseed)
    average_r=0.0
    average_r_pearson=0.0
    average_rmse=0.0
    if CV=='kf':
        kf = KFold(n_splits=k_fold,shuffle=True,random_state=iseed)
        validation=kf.split(X)
    if CV=='loo':
        loo = LeaveOneOut()
        validation=loo.split(X)
    if CV=='kf' or CV=='loo':
        real_y=[]
        predicted_y=[]
        counter=1
        for train_index, test_index in validation:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            GBR = GradientBoostingRegressor(criterion=GBR_criterion,n_estimators=GBR_n_estimators,learning_rate=GBR_learning_rate,max_depth=GBR_max_depth,min_samples_split=GBR_min_samples_split,min_samples_leaf=GBR_min_samples_leaf)
            y_pred = GBR.fit(X_train, y_train).predict(X_test)
            for i in range(len(y_test)):
                #f_out.write("y_test[i] %s \n" %(str(y_test[i])))
                real_y.append(y_test[i])
            for i in range(len(y_pred)):
                #f_out.write("y_pred[i] %s \n" %(str(y_pred[i])))
                predicted_y.append(y_pred[i])
            if CV=='kf':
                r_pearson,_=pearsonr(y_test,y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                if verbose_level>=2: f_out.write('Landscape %i . Adventurousness: %i . k-fold: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],counter,r_pearson,rmse))
                if verbose_level>=2: f_out.write("%i test points: %s \n" % (len(test_index),str(test_index)))
                counter=counter+1
                average_r_pearson=average_r_pearson+r_pearson
                average_rmse=average_rmse+rmse
        if CV=='kf':
            average_r_pearson=average_r_pearson/k_fold
            average_rmse=average_rmse/k_fold
            if verbose_level>=2: f_out.write('k-fold average r_pearson score: %f \n' % (average_r_pearson))
            if verbose_level>=2: f_out.write('k-fold average rmse score: %f \n' % (average_rmse))
        total_r_pearson,_ = pearsonr(real_y,predicted_y)
        total_mse = mean_squared_error(real_y, predicted_y)
        total_rmse = np.sqrt(total_mse)
    elif CV=='sort':
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_last_percentage,random_state=iseed,shuffle=False)
        GBR = GradientBoostingRegressor(criterion=GBR_criterion,n_estimators=GBR_n_estimators,learning_rate=GBR_learning_rate,max_depth=GBR_max_depth,min_samples_split=GBR_min_samples_split,min_samples_leaf=GBR_min_samples_leaf)
        y_pred = GBR.fit(X_train, y_train).predict(X_test)
        total_r_pearson,_=pearsonr(y_test,y_pred)
        mse = mean_squared_error(y_test, y_pred)
        total_rmse = np.sqrt(mse)
        if  verbose_level>=1: f_out.write("Train with first %i points \n" % (len(X_train)))
        if  verbose_level>=1: f_out.write("%s \n" % (str(X_train)))
        if  verbose_level>=1: f_out.write("Test with last %i points \n" % (len(X_test)))
        if  verbose_level>=1: f_out.write("%s \n" % (str(X_test)))
        if verbose_level>=1: f_out.write('Landscape %i . Adventurousness: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],total_r_pearson,total_rmse))
    if verbose_level>=1: f_out.write('Final r_pearson score: %f \n' % (total_r_pearson))
    if verbose_level>=1: f_out.write('Final rmse score: %f \n' % (total_rmse))
    if error_metric=='rmse': result=total_rmse
    if verbose_level>=1: f_out.flush()
    return result


# CALCULATE GPR #
def GPR(X,y,iseed,l,w,f_out,Xtr,ytr,mode):
    iseed=iseed+1
    # Calculate error metric
    if mode==1:
        if verbose_level>=1: f_out.write('## Start: "GPR" function \n')
        if verbose_level>=1: f_out.write('-------- \n')
        if verbose_level>=1: f_out.write('Perform GPR\n')
        if verbose_level>=1: f_out.write('Cross_validation %i - fold\n' % (k_fold))
        if verbose_level>=1: f_out.write('Initial A_RBF %f: \n' % (A_RBF))
        if verbose_level>=1: f_out.write('Initial kernel_length_scale: %f: \n' % (kernel_length_scale))
        if verbose_level>=1: f_out.write('Initial kernel_noise_level: %f: \n' % (kernel_noise_level))
        if verbose_level>=1: f_out.write('-------- \n')
        if verbose_level>=1: f_out.flush()
        average_r=0.0
        average_r_pearson=0.0
        average_rmse=0.0
        if CV=='kf':
            kf = KFold(n_splits=k_fold,shuffle=True,random_state=iseed)
            validation=kf.split(X)
        if CV=='loo':
            loo = LeaveOneOut()
            validation=loo.split(X)
        if CV=='kf' or CV=='loo':
            real_y=[]
            predicted_y=[]
            counter=0
            for train_index, test_index in validation:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                kernel = A_RBF * RBF(length_scale=kernel_length_scale, length_scale_bounds=(1e-3, 1e+3)) + A_noise * WhiteKernel(noise_level=kernel_noise_level, noise_level_bounds=(1e-5, 1e+1))
                GPR = GaussianProcessRegressor(kernel=kernel,alpha=GPR_alpha,normalize_y=True)
                #y_pred = GPR.fit(X_train, y_train).predict(X_test)
                y_pred = GPR.fit(X_train_scaled, y_train).predict(X_test_scaled)
                if verbose_level>=2: f_out.write('TEST X_train: \n')
                if verbose_level>=2: f_out.write('%s \n' % (str(X_train)))
                if verbose_level>=2: f_out.write('TEST y_train: \n')
                if verbose_level>=2: f_out.write('%s \n' % (str(y_train)))
                if verbose_level>=2: f_out.write('TEST X_test: \n')
                if verbose_level>=2: f_out.write('%s \n' % (str(X_test)))
                if verbose_level>=2: f_out.write('TEST y_test: \n')
                if verbose_level>=2: f_out.write('%s \n' % (str(y_test)))
                if verbose_level>=2: f_out.write('Converged kernel hyperparameters: %s \n' % (str(GPR.kernel_)))
                if verbose_level>=2: f_out.write('Converged alpha: %s \n' % (str(GPR.alpha_)))
                if verbose_level>=2: f_out.write('Parameters GPR: \n')
                if verbose_level>=2: f_out.write('%s \n' % (str(GPR.get_params(deep=True))))
                if verbose_level>=2: f_out.write('Parameters GPR kernel: \n')
                if verbose_level>=2: f_out.write('%s \n' % (str(GPR.kernel_.get_params(deep=True))))
                if verbose_level>=2: f_out.write('GPR X_train: \n')
                if verbose_level>=2: f_out.write('%s \n' % (str(GPR.X_train_)))
                if verbose_level>=2: f_out.write('GPR y_train: \n')
                if verbose_level>=2: f_out.write('%s \n' % (str(GPR.y_train_)))
                if verbose_level>=2: f_out.write('TEST X_train_scaled: \n')
                if verbose_level>=2: f_out.write('%s \n' % (str(X_train_scaled)))
                if verbose_level>=2: f_out.write('TEST X_test_scaled: \n')
                if verbose_level>=2: f_out.write('%s \n' % (str(X_test_scaled)))
                if verbose_level>=2: f_out.write('TEST y_test: \n')
                if verbose_level>=2: f_out.write('%s \n' % (str(y_test)))
                if verbose_level>=2: f_out.write('TEST y_pred: \n')
                if verbose_level>=2: f_out.write('%s \n' % (str(y_pred)))
                if verbose_level>=1: f_out.flush()
    
                for i in range(len(y_test)):
                    #f_out.write("y_test[i] %s \n" %(str(y_test[i])))
                    real_y.append(y_test[i])
                for i in range(len(y_pred)):
                    #f_out.write("y_pred[i] %s \n" %(str(y_pred[i])))
                    predicted_y.append(y_pred[i])
                if CV=='kf':
                    r_pearson,_=pearsonr(y_test,y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    if verbose_level>=2: f_out.write('Landscape %i . Adventurousness: %i . k-fold: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],counter,r_pearson,rmse))
                    if verbose_level>=2: f_out.write("%i test points: %s \n" % (len(test_index),str(test_index)))
                    counter=counter+1
                    average_r_pearson=average_r_pearson+r_pearson
                    average_rmse=average_rmse+rmse
                if CV=='kf':
                    r_pearson,_=pearsonr(y_test,y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    if verbose_level>=2: f_out.write('Landscape %i . Adventurousness: %i . k-fold: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],counter,r_pearson,rmse))
                    if verbose_level>=2: f_out.write("%i test points: %s \n" % (len(test_index),str(test_index)))
                    counter=counter+1
                    average_r_pearson=average_r_pearson+r_pearson
                    average_rmse=average_rmse+rmse
            if CV=='kf':
                average_r_pearson=average_r_pearson/k_fold
                average_rmse=average_rmse/k_fold
                if verbose_level>=2: f_out.write('k-fold average r_pearson score: %f \n' % (average_r_pearson))
                if verbose_level>=2: f_out.write('k-fold average rmse score: %f \n' % (average_rmse))
            total_r_pearson,_ = pearsonr(real_y,predicted_y)
            total_mse = mean_squared_error(real_y, predicted_y)
            total_rmse = np.sqrt(total_mse)
        elif CV=='sort':
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_last_percentage,random_state=iseed,shuffle=False)
            kernel = A_RBF * RBF(length_scale=kernel_length_scale, length_scale_bounds=(1e-3, 1e+3)) + A_noise * WhiteKernel(noise_level=kernel_noise_level, noise_level_bounds=(1e-5, 1e+1))
            GPR = GaussianProcessRegressor(kernel=kernel,alpha=GPR_alpha,normalize_y=True)
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            y_pred = GPR.fit(X_train_scaled, y_train).predict(X_test_scaled)
            if verbose_level>=2: f_out.write('TEST X_train: \n')
            if verbose_level>=2: f_out.write('%s \n' % (str(X_train)))
            if verbose_level>=2: f_out.write('TEST y_train: \n')
            if verbose_level>=2: f_out.write('%s \n' % (str(y_train)))
            if verbose_level>=2: f_out.write('TEST X_test: \n')
            if verbose_level>=2: f_out.write('%s \n' % (str(X_test)))
            if verbose_level>=2: f_out.write('TEST y_test: \n')
            if verbose_level>=2: f_out.write('%s \n' % (str(y_test)))
            if verbose_level>=2: f_out.write('Converged kernel hyperparameters: %s \n' % (str(GPR.kernel_)))
            if verbose_level>=2: f_out.write('Converged alpha: %s \n' % (str(GPR.alpha_)))
            if verbose_level>=2: f_out.write('Parameters GPR: \n')
            if verbose_level>=2: f_out.write('%s \n' % (str(GPR.get_params(deep=True))))
            if verbose_level>=2: f_out.write('Parameters GPR kernel: \n')
            if verbose_level>=2: f_out.write('%s \n' % (str(GPR.kernel_.get_params(deep=True))))
            if verbose_level>=2: f_out.write('GPR X_train: \n')
            if verbose_level>=2: f_out.write('%s \n' % (str(GPR.X_train_)))
            if verbose_level>=2: f_out.write('GPR y_train: \n')
            if verbose_level>=2: f_out.write('%s \n' % (str(GPR.y_train_)))
            if verbose_level>=2: f_out.write('TEST X_train_scaled: \n')
            if verbose_level>=2: f_out.write('%s \n' % (str(X_train_scaled)))
            if verbose_level>=2: f_out.write('TEST X_test_scaled: \n')
            if verbose_level>=2: f_out.write('%s \n' % (str(X_test_scaled)))
            if verbose_level>=2: f_out.write('TEST y_test: \n')
            if verbose_level>=2: f_out.write('%s \n' % (str(y_test)))
            if verbose_level>=2: f_out.write('TEST y_pred: \n')
            if verbose_level>=2: f_out.write('%s \n' % (str(y_pred)))
            if verbose_level>=1: f_out.flush()
    
            total_r_pearson,_=pearsonr(y_test,y_pred)
            mse = mean_squared_error(y_test, y_pred)
            total_rmse = np.sqrt(mse)
            if  verbose_level>=1: f_out.write("Train with first %i points \n" % (len(X_train)))
            if  verbose_level>=1: f_out.write("%s \n" % (str(X_train)))
            if  verbose_level>=1: f_out.write("Test with last %i points \n" % (len(X_test)))
            if  verbose_level>=1: f_out.write("%s \n" % (str(X_test)))
            if verbose_level>=1: f_out.write('Landscape %i . Adventurousness: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],total_r_pearson,total_rmse))
        if verbose_level>=1: f_out.write('Final r_pearson score: %f \n' % (total_r_pearson))
        if verbose_level>=1: f_out.write('Final rmse score: %f \n' % (total_rmse))
        if verbose_level>=1: f_out.flush()
        if error_metric=='rmse': result=total_rmse
    if mode==2:
        if verbose_level>=1: f_out.write('## Start: "GPR" function \n')
        if verbose_level>=1: f_out.write('-------- \n')
        if verbose_level>=1: f_out.write('Perform GPR\n')
        if verbose_level>=1: f_out.write('Initial A_RBF %f \n' % (A_RBF))
        if verbose_level>=1: f_out.write('Initial kernel_length_scale: %f \n' % (kernel_length_scale))
        if verbose_level>=1: f_out.write('Initial kernel_noise_level: %f \n' % (kernel_noise_level))
        if verbose_level>=1: f_out.write('-------- \n')
        if verbose_level>=1: f_out.flush()

        X_train, X_test = Xtr, X
        y_train, y_test = ytr, y
        real_y=[]
        predicted_y=[]
        result = []

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        kernel = A_RBF * RBF(length_scale=kernel_length_scale, length_scale_bounds=(1e-3, 1e+3)) + A_noise * WhiteKernel(noise_level=kernel_noise_level, noise_level_bounds=(1e-5, 1e+1))
        GPR = GaussianProcessRegressor(kernel=kernel,alpha=GPR_alpha,normalize_y=True)
        GPR.fit(X_train_scaled,y_train)
        y_pred = GPR.predict(X_test_scaled)
        if verbose_level>=2: f_out.write('TEST X_train: \n')
        if verbose_level>=2: f_out.write('%s \n' % (str(X_train)))
        if verbose_level>=2: f_out.write('TEST y_train: \n')
        if verbose_level>=2: f_out.write('%s \n' % (str(y_train)))
        if verbose_level>=2: f_out.write('TEST X_test: \n')
        if verbose_level>=2: f_out.write('%s \n' % (str(X_test)))
        if verbose_level>=2: f_out.write('TEST y_test: \n')
        if verbose_level>=2: f_out.write('%s \n' % (str(y_test)))
        if verbose_level>=1: f_out.write('Converged kernel hyperparameters: %s \n' % (str(GPR.kernel_)))
        if verbose_level>=2: f_out.write('Converged alpha: %s \n' % (str(GPR.alpha_)))
        if verbose_level>=2: f_out.write('Parameters GPR: \n')
        if verbose_level>=2: f_out.write('%s \n' % (str(GPR.get_params(deep=True))))
        if verbose_level>=2: f_out.write('Parameters GPR kernel: \n')
        if verbose_level>=2: f_out.write('%s \n' % (str(GPR.kernel_.get_params(deep=True))))
        if verbose_level>=2: f_out.write('GPR X_train: \n')
        if verbose_level>=2: f_out.write('%s \n' % (str(GPR.X_train_)))
        if verbose_level>=2: f_out.write('GPR y_train: \n')
        if verbose_level>=2: f_out.write('%s \n' % (str(GPR.y_train_)))
        if verbose_level>=2: f_out.write('TEST X_train_scaled: \n')
        if verbose_level>=2: f_out.write('%s \n' % (str(X_train_scaled)))
        if verbose_level>=2: f_out.write('TEST X_test_scaled: \n')
        if verbose_level>=2: f_out.write('%s \n' % (str(X_test_scaled)))
        if verbose_level>=2: f_out.write('TEST y_test: \n')
        if verbose_level>=2: f_out.write('%s \n' % (str(y_test)))
        if verbose_level>=2: f_out.write('TEST y_pred: \n')
        if verbose_level>=2: f_out.write('%s \n' % (str(y_pred)))
        if verbose_level>=2: f_out.flush()
        for i in range(len(y_test)):
            #f_out.write("y_test[i] %s \n" %(str(y_test[i])))
            real_y.append(y_test[i])
        for i in range(len(y_pred)):
            #f_out.write("y_pred[i] %s \n" %(str(y_pred[i])))
            predicted_y.append(y_pred[i])
        min_index = predicted_y.index(min(predicted_y))
        if verbose_level>=2: f_out.write("At index %i, predicted minimum value: %f\n" %(min_index, min(predicted_y)))
        if verbose_level>=2: f_out.write("At index %i, real minimum value: %f\n" %(min_index, min(real_y)))
        for j in range(param):
            result.append(X_test[min_index][j])
        result.append(min(predicted_y))
    return result

# CALCULATE KRR #
def KRR(hyperparams,X,y,iseed,l,w,f_out,verbose_level):
    #KRR_alpha, KRR_gamma = hyperparams
    KRR_gamma = hyperparams
    iseed=iseed+1
    if verbose_level>=1: f_out.write('## Start: "KRR" function \n')
    if verbose_level>=1: f_out.write('-------- \n')
    if verbose_level>=1: f_out.write('Perform KRR\n')
    if verbose_level>=1: f_out.write('Cross_validation %i - fold\n' % (k_fold))
    if verbose_level>=1: f_out.write('KRR alpha %f\n' % (KRR_alpha))
    if verbose_level>=1: f_out.write('KRR gamma %f\n' % (KRR_gamma))
    if verbose_level>=1: f_out.write('KRR kernel %s\n' % (KRR_kernel))
    if verbose_level>=1: f_out.write('-------- \n')
    if verbose_level>=1: f_out.flush()
    average_r=0.0
    average_r_pearson=0.0
    average_rmse=0.0
    if CV=='kf':
        kf = KFold(n_splits=k_fold,shuffle=True,random_state=iseed)
        validation=kf.split(X)
    if CV=='loo':
        loo = LeaveOneOut()
        validation=loo.split(X)
    if CV=='kf' or CV=='loo':
        real_y=[]
        predicted_y=[]
        counter=0
        for train_index, test_index in validation:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            KRR = KernelRidge(alpha=KRR_alpha,kernel=KRR_kernel,gamma=KRR_gamma)
            y_pred = KRR.fit(X_train, y_train).predict(X_test)
            #if verbose_level>=1: f_out.write('Parameters KRR: \n')
            #if verbose_level>=1: f_out.write('%s \n' % (str(KRR.get_params(deep=True))))

            for i in range(len(y_test)):
                #f_out.write("y_test[i] %s \n" %(str(y_test[i])))
                real_y.append(y_test[i])
            for i in range(len(y_pred)):
                #f_out.write("y_pred[i] %s \n" %(str(y_pred[i])))
                predicted_y.append(y_pred[i]) #
            if CV=='kf':
                r_pearson,_=pearsonr(y_test,y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                if verbose_level>=2: f_out.write('Landscape %i . Adventurousness: %i . k-fold: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],counter,r_pearson,rmse))
                if verbose_level>=2: f_out.write("%i test points: %s \n" % (len(test_index),str(test_index)))
                counter=counter+1
                average_r_pearson=average_r_pearson+r_pearson
                average_rmse=average_rmse+rmse
        if CV=='kf':
            average_r_pearson=average_r_pearson/k_fold
            average_rmse=average_rmse/k_fold
            if verbose_level>=2: f_out.write('k-fold average r_pearson score: %f \n' % (average_r_pearson))
            if verbose_level>=2: f_out.write('k-fold average rmse score: %f \n' % (average_rmse))
        total_r_pearson,_ = pearsonr(real_y,predicted_y)
        total_mse = mean_squared_error(real_y, predicted_y)
        total_rmse = np.sqrt(total_mse)
    elif CV=='sort':
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_last_percentage,random_state=iseed,shuffle=False)
        KRR = KernelRidge(alpha=KRR_alpha,kernel=KRR_kernel,gamma=KRR_gamma)
        y_pred = KRR.fit(X_train, y_train).predict(X_test)
        total_r_pearson,_=pearsonr(y_test,y_pred)
        mse = mean_squared_error(y_test, y_pred)
        total_rmse = np.sqrt(mse)
        if  verbose_level>=1: f_out.write("Train with first %i points \n" % (len(X_train)))
        if  verbose_level>=1: f_out.write("%s \n" % (str(X_train)))
        if  verbose_level>=1: f_out.write("Test with last %i points \n" % (len(X_test)))
        if  verbose_level>=1: f_out.write("%s \n" % (str(X_test)))
        if verbose_level>=1: f_out.write('Landscape %i . Adventurousness: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],total_r_pearson,total_rmse))
    if verbose_level>=1: f_out.write('Parameters KRR: \n')
    if verbose_level>=1: f_out.write('%s \n' % (str(KRR.get_params(deep=True))))
    if verbose_level>=1: f_out.write('Final r_pearson score: %f \n' % (total_r_pearson))
    if verbose_level>=1: f_out.write('Final rmse score: %f \n' % (total_rmse))
    if error_metric=='rmse': result=total_rmse
    if verbose_level>=1: f_out.flush()
    return result

# PLOT #
def plot(flag,final_result_T):
    if flag=='rmse':
        pntbox=plt.boxplot(final_result_T,patch_artist=True,labels=adven,showfliers=False)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel('Adventurousness (%)',fontsize=15)
        plt.ylabel('RMSE (arb. units)',fontsize=15)
        file1='rmse.png'
        plt.savefig(file1,format='png',dpi=600)
        print('save rmse box plot to %s' %file1,flush=True)
        plt.close()

        pntbox=plt.boxplot(final_result_T,patch_artist=True,labels=adven,showfliers=True)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel('Adventurousness (%)',fontsize=15)
        plt.ylabel('RMSE (arb. units)',fontsize=15)
        file1='rmse_with_outliers.png'
        plt.savefig(file1,format='png',dpi=600)
        print('save rmse box plot to %s' %file1,flush=True)
        plt.close()

##### END OTHER FUNCTIONS ######
################################################################################
start = time()
(is_dask,NCPU,verbose_level,log_name,Nspf,S,iseed,param,center_min,center_max,grid_min,grid_max,grid_Delta,Nwalkers,adven,t1_time,d_threshold,t0_time,initial_sampling,ML,error_metric,CV,k_fold,test_last_percentage,n_neighbor,weights,GBR_criterion,GBR_n_estimators,GBR_learning_rate,GBR_max_depth,GBR_min_samples_split,GBR_min_samples_leaf,A_RBF,A_noise,GPR_alpha,kernel_length_scale,kernel_noise_level,KRR_alpha,KRR_kernel,KRR_gamma,optimize_gamma,KRR_gamma_lim,allowed_initial_sampling,allowed_CV,allowed_ML,allowed_ML,allowed_error_metric,width_min,width_max,Amplitude_min,Amplitude_max,N,t2_time,allowed_verbosity_level,t2_ML,allowed_t2_ML) = read_initial_values(input_file_name)
main(iseed)
time_taken = time()-start
print ('Process took %0.2f seconds' %time_taken)
