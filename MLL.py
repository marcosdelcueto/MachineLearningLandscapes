#!/usr/bin/env python3
# Marcos del Cueto
# Department of Chemistry and MIF, University of Liverpool
#################################################################################
import ast
import sys
import dask
import random
import math
import copy
import statistics
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from time import time
from dask import delayed
from pickle import dump,load
from sklearn import neighbors
from numpy.random import choice
from sklearn import preprocessing
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from scipy.optimize import differential_evolution
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

#################################################################################
######   START CUSTOMIZABLE PARAMETERS ########
input_file_name = 'input_MLL.inp'      # name of input file
######   END CUSTOMIZABLE PARAMETERS   ######
#################################################################################

#################################################################################
###### START MAIN ######
def main(iseed):
    # Check that input values are OK
    check_input_values()
    # Calculation just to generate SPF grid
    if calculate_grid == True:
        # Use paralellization: a SPF per CPU
        if dask_parallel==True:
            dummy_list=[]
            for l in range(Nspf):
                iseed=iseed+l
                dummy = delayed(generate_grid)(iseed,l)
                dummy_list.append(dummy)
            result=dask.compute(dummy_list,scheduler='processes',num_workers=NCPU)
        # Calculate SPFs in serial (all in 1 CPU)
        if dask_parallel==False:
            for l in range(Nspf):
                iseed=iseed+l
                Ngrid = generate_grid(iseed,l)

    # Calculation to explore SPF
    #elif calculate_grid == False:
    if t1_analysis == True:
        # Initialize results array
        results_t1_per_Nspf=[]
        results_t2_per_Nspf=[]
        # Calculate results for each landscape (may use dask to run each landscape in a CPU in parallel)
        if dask_parallel==True:
            for l in range(initial_spf,initial_spf+Nspf):
                iseed=iseed+l
                (provi_result_t1,provi_result_t2)=delayed(MLL,nout=2)(iseed,l)
                results_t1_per_Nspf.append(provi_result_t1)
                results_t2_per_Nspf.append(provi_result_t2)
            results_t1_per_Nspf=dask.compute(results_t1_per_Nspf,scheduler='processes',num_workers=NCPU)
            results_t2_per_Nspf=dask.compute(results_t2_per_Nspf,scheduler='processes',num_workers=NCPU)
            results_t1_per_Nspf=results_t1_per_Nspf[0]
            results_t2_per_Nspf=results_t2_per_Nspf[0]
        elif dask_parallel==False:
            for l in range(initial_spf,initial_spf+Nspf):
                iseed=iseed+l
                (provi_result_t1,provi_result_t2)=MLL(iseed,l)
                results_t1_per_Nspf.append(provi_result_t1)
                results_t2_per_Nspf.append(provi_result_t2)
        # Transpose results_per_Nspf, to get results per walker
        if t1_analysis    == True: results_per_walker_t1=[list(i) for i in zip(*results_t1_per_Nspf)]
        if t2_exploration == True: results_per_walker_t2=[list(i) for i in zip(*results_t2_per_Nspf)]
        # Print final results
        print('--- Final results ---')
        for i in range(Nwalkers):
            print('-- Adventurousness: %6.1f --' %(adven[i]))
            if t1_analysis == True:
                print('-- t1 analysis')
                print('- RMSE:',results_per_walker_t1[i][:])
                print('- RMSE Mean: %f' %(statistics.mean(results_per_walker_t1[i])))
                print('- RMSE Median: %f' %(statistics.median(results_per_walker_t1[i])))
            if t2_exploration == True:
                print('-- t2 exploration')
                print('- [ML_gain_pred, ML_gain_real, error_rel_ML, min_standard, min_MLML_gain_real_relative]: %s' %(str(results_per_walker_t2[i])))
                ML_gain_pred          = [item[0] for item in results_per_walker_t2[i]]
                ML_gain_real          = [item[1] for item in results_per_walker_t2[i]]
                error_rel_ML          = [item[2] for item in results_per_walker_t2[i]]
                min_standard          = [item[3] for item in results_per_walker_t2[i]]
                min_ML                = [item[4] for item in results_per_walker_t2[i]]
                ML_gain_real_relative = [item[5] for item in results_per_walker_t2[i]]
                print('- ML_gain_pred Mean: %f' %(statistics.mean(ML_gain_pred)))
                print('- ML_gain_pred Median: %f' %(statistics.median(ML_gain_pred)))
                print('- ML_gain_real Mean: %f' %(statistics.mean(ML_gain_real)))
                print('- ML_gain_real Median: %f' %(statistics.median(ML_gain_real)))
                print('- error_rel_ML Mean: %f' %(statistics.mean(error_rel_ML)))
                print('- error_rel_ML Median: %f' %(statistics.median(error_rel_ML)))
                print('- min_standard Mean: %f' %(statistics.mean(min_standard)))
                print('- min_standard Median: %f' %(statistics.median(min_standard)))
                print('- min_ML Mean: %f' %(statistics.mean(min_ML)))
                print('- min_ML Median: %f' %(statistics.median(min_ML)))
                print('- ML_gain_real_relative Mean: %f' %(statistics.mean(ML_gain_real_relative)))
                print('- ML_gain_real_relative Median: %f' %(statistics.median(ML_gain_real_relative)))
            print('',flush=True)
        if plot_t1_error_metric == True and error_metric=='rmse':
            plot(error_metric,None,None,None,None,None,None,None,None,None,results_per_walker_t1)
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
    dask_parallel = ast.literal_eval(var_value[var_name.index('dask_parallel')])
    NCPU = ast.literal_eval(var_value[var_name.index('NCPU')])
    verbosity_level = ast.literal_eval(var_value[var_name.index('verbosity_level')])
    log_name = ast.literal_eval(var_value[var_name.index('log_name')])
    Nspf = ast.literal_eval(var_value[var_name.index('Nspf')])
    initial_spf = ast.literal_eval(var_value[var_name.index('initial_spf')])
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
    GPR_alpha = ast.literal_eval(var_value[var_name.index('GPR_alpha')])
    GPR_length_scale = ast.literal_eval(var_value[var_name.index('GPR_length_scale')])
    optimize_GPR_hyperparams = ast.literal_eval(var_value[var_name.index('optimize_GPR_hyperparams')])
    GPR_alpha_lim = ast.literal_eval(var_value[var_name.index('GPR_alpha_lim')])
    GPR_length_scale_lim = ast.literal_eval(var_value[var_name.index('GPR_length_scale_lim')])
    #GPR_n_restarts_optimizer = ast.literal_eval(var_value[var_name.index('GPR_n_restarts_optimizer')])
    #GPR_A_RBF = ast.literal_eval(var_value[var_name.index('GPR_A_RBF')])
    #GPR_noise_level = ast.literal_eval(var_value[var_name.index('GPR_noise_level')])
    KRR_alpha = ast.literal_eval(var_value[var_name.index('KRR_alpha')])
    KRR_kernel = ast.literal_eval(var_value[var_name.index('KRR_kernel')])
    KRR_gamma = ast.literal_eval(var_value[var_name.index('KRR_gamma')])
    optimize_KRR_hyperparams = ast.literal_eval(var_value[var_name.index('optimize_KRR_hyperparams')])
    KRR_alpha_lim = ast.literal_eval(var_value[var_name.index('KRR_alpha_lim')])
    KRR_gamma_lim = ast.literal_eval(var_value[var_name.index('KRR_gamma_lim')])
    allowed_initial_sampling = ast.literal_eval(var_value[var_name.index('allowed_initial_sampling')])
    allowed_CV = ast.literal_eval(var_value[var_name.index('allowed_CV')])
    allowed_ML = ast.literal_eval(var_value[var_name.index('allowed_ML')])
    allowed_error_metric = ast.literal_eval(var_value[var_name.index('allowed_error_metric')])
    t2_time = ast.literal_eval(var_value[var_name.index('t2_time')])
    allowed_verbosity_level = ast.literal_eval(var_value[var_name.index('allowed_verbosity_level')])
    t2_ML = ast.literal_eval(var_value[var_name.index('t2_ML')])
    allowed_t2_ML = ast.literal_eval(var_value[var_name.index('allowed_t2_ML')])
    t2_exploration = ast.literal_eval(var_value[var_name.index('t2_exploration')])
    t1_analysis = ast.literal_eval(var_value[var_name.index('t1_analysis')])
    diff_popsize = ast.literal_eval(var_value[var_name.index('diff_popsize')])
    diff_tol = ast.literal_eval(var_value[var_name.index('diff_tol')])
    t2_train_time = ast.literal_eval(var_value[var_name.index('t2_train_time')])
    calculate_grid = ast.literal_eval(var_value[var_name.index('calculate_grid')])
    plot_t1_exploration = ast.literal_eval(var_value[var_name.index('plot_t1_exploration')])
    plot_t1_error_metric = ast.literal_eval(var_value[var_name.index('plot_t1_error_metric')])
    plot_contour_map = ast.literal_eval(var_value[var_name.index('plot_contour_map')])
    grid_name = ast.literal_eval(var_value[var_name.index('grid_name')])

    width_min=S                                     # Minimum width of each Gaussian function
    width_max=1.0/3.0                               # Maximum width of each Gaussian function
    Amplitude_min=0.0                               # Minimum amplitude of each Gaussian function
    Amplitude_max=1.0                               # Maximum amplitude of each Gaussian function
    N=int(round((1/(S**param))))                    # Number of Gaussian functions of a specific landscape
    if iseed==None: 
        iseed=random.randrange(2**30-1) # If no seed is specified, choose a random one

    return (dask_parallel, NCPU, verbosity_level, log_name, Nspf, S, iseed, param, center_min, center_max, grid_min, grid_max, grid_Delta, Nwalkers, adven, t1_time, d_threshold, t0_time, initial_sampling, ML, error_metric, CV, k_fold, test_last_percentage, n_neighbor, weights, GBR_criterion, GBR_n_estimators, GBR_learning_rate, GBR_max_depth, GBR_min_samples_split, GBR_min_samples_leaf, GPR_alpha, GPR_length_scale, GPR_alpha_lim , GPR_length_scale_lim, KRR_alpha, KRR_kernel, KRR_gamma, optimize_KRR_hyperparams, optimize_GPR_hyperparams, KRR_alpha_lim, KRR_gamma_lim, allowed_initial_sampling, allowed_CV, allowed_ML, allowed_ML, allowed_error_metric, width_min, width_max, Amplitude_min, Amplitude_max, N, t2_time, allowed_verbosity_level, t2_ML, allowed_t2_ML, t2_exploration, t1_analysis, diff_popsize, diff_tol, t2_train_time, calculate_grid, grid_name,plot_t1_exploration,plot_contour_map,plot_t1_error_metric,initial_spf)

def MLL(iseed,l):
    # open log file to write intermediate information
    if verbosity_level>=1:
        f_out = open('%s_%s.log' % (log_name,l), 'w')
    else:
        f_out=None
    # initialize result arrays
    result1=None
    result2=None
    error_metric_list=[]
    ML_benefits_list=[]
    # Generate SPF grid
    time_taken1 = time()-start
    # Old: generate grid
    #dim_list, G_list, Ngrid, max_G = generate_grid(iseed,l,f_out)
    #  New: read grid from file
    filename = grid_name + '_' + str(l) + '.log'
    dim_list       = [[] for i in range(param)]
    G_list         = []
    with open(filename) as file_in:
        for line in file_in:
            counter_word=-1
            try:
                if isinstance(int(line.split()[0]),int):
                    for word in line.split():
                        if counter_word >= 0 and counter_word < param:  dim_list[counter_word].append(float(word))
                        if counter_word == param: G_list.append(float(word))
                        counter_word=counter_word+1
            except:
                pass
    Ngrid=int((grid_max/grid_Delta+1)**param)   # calculate number of grid points
    max_G=max(G_list)
    min_G=min(G_list)
    #for i in range(len(G_list)):
        #print(i,dim_list[0][i], dim_list[1][i],G_list[i])
    ##################################
    time_taken2 = time()-start
    if verbosity_level>=1:
        f_out.write("Generate grid took %0.4f seconds\n" %(time_taken2-time_taken1))
    # For each walker
    for w in range(Nwalkers):
        # Step 1) Perform t1 exploration
        time_taken1 = time()-start
        #X0,y0,unique_t0 = explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,t0_time,0,0,None,None,False)
        X0,y0,unique_t0 = explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,t0_time,0,0,None,None,False,None,None)
        #print('TEST X0:')
        #print(X0)
        #print('TEST y0:')
        #print(y0)
        #X1,y1,unique_t1 = explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,t0_time,t1_time,0,None,None,False)
        X1,y1,unique_t1 = explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,0,t1_time,0,None,None,False,X0,y0)
        #print('TEST after X0:')
        #print(X0)
        #print('TEST after y0:')
        #print(y0)
        #print('TEST X1:')
        #print(X1)
        #print('TEST y1:')
        #print(y1)
        time_taken2 = time()-start
        if verbosity_level>=1:
            f_out.write("t1 exploration took %0.4f seconds\n" %(time_taken2-time_taken1))
        if t1_analysis == True:
        # Step 2A) Calculate error_metric
            time_taken1 = time()-start
            if ML=='kNN': error_metric_result=kNN(X1,y1,iseed,l,w,f_out,None,None,1,None)
            if ML=='GBR': error_metric_result=GBR(X1,y1,iseed,l,w,f_out)
            #if ML=='GPR': error_metric_result=GPR1(X1,y1,iseed,l,w,f_out,None,None,1,None)
            if ML=='GPR':
                hyperparams=[GPR_alpha,GPR_length_scale]
                if optimize_GPR_hyperparams == False:
                    error_metric_result=GPR(hyperparams,X1,y1,iseed,l,w,f_out,None,None,1,None)
                else:
                    mini_args=(X1,y1,iseed,l,w,f_out,None,None,1,None)
                    bounds = [GPR_alpha_lim]+[GPR_length_scale_lim]
                    solver=differential_evolution(GPR,bounds,args=mini_args,popsize=diff_popsize,tol=diff_tol)
                    best_hyperparams = solver.x
                    best_rmse = solver.fun
                    if verbosity_level>=1: 
                        f_out.write("Best hyperparameters: %s \n" %str(best_hyperparams))
                        f_out.write("Best rmse: %f \n"  %best_rmse)
                        f_out.flush()
                    error_metric_result = best_rmse
            if ML=='KRR':
                hyperparams=[KRR_alpha,KRR_gamma]
                if optimize_KRR_hyperparams == False:
                    error_metric_result=KRR(hyperparams,X1,y1,iseed,l,w,f_out,None,None,1,None)
                else:
                    mini_args=(X1,y1,iseed,l,w,f_out,None,None,1,None)
                    bounds = [KRR_alpha_lim]+[KRR_gamma_lim]
                    solver=differential_evolution(KRR,bounds,args=mini_args,popsize=diff_popsize,tol=diff_tol)
                    best_hyperparams = solver.x
                    best_rmse = solver.fun
                    if verbosity_level>=1: 
                        f_out.write("Best hyperparameters: %s \n" %str(best_hyperparams))
                        f_out.write("Best rmse: %f \n"  %best_rmse)
                        f_out.flush()
                    error_metric_result = best_rmse
            error_metric_list.append(error_metric_result)
            result1 = error_metric_list
            time_taken2 = time()-start
            ###################################
            ###################################
            ###################################
            # TEST plot 2d exploration
            if plot_t1_exploration == True and param == 2:
                plot('t1_exploration',l,w,iseed,dim_list,G_list,X0,y0,X1,y1,None)
            ###################################
            ###################################
            ###################################
            if verbosity_level>=1:
                f_out.write("t1 analysis took %0.4f seconds\n" %(time_taken2-time_taken1))
        # Step 2B) Perform t2 exploration
        if t2_exploration == True:
            # Step 2.B.1) Perform t2 exploration with random biased explorer
            time_taken1 = time()-start
            #X2a,y2a,unique_t2a = explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,0,unique_t1,t2_time,X1,y1,False)
            X2a,y2a,unique_t2a = explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,0,unique_t1,t2_time,X1,y1,False,None,None)
            time_taken2 = time()-start
            if verbosity_level>=1:
                f_out.write("t2 standard exploration took %0.4f seconds\n" %(time_taken2-time_taken1))
            # Step 2.B.2) Perform t2 exploration with ML explorer
            time_taken1 = time()-start
            #X2b,y2b,unique_t2b = explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,0,unique_t1,t2_time,X1,y1,True)
            X2b,y2b,unique_t2b = explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,0,unique_t1,t2_time,X1,y1,True,None,None)
            time_taken2 = time()-start
            if verbosity_level>=1:
                f_out.write("t2 ML exploration took %0.4f seconds\n" %(time_taken2-time_taken1))
            # select coordinates where minimum is predicted with ML
            time_taken1 = time()-start
            x_real=[]
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
                            k_in_grid=new_k_in_grid[0][0][i]
            if verbosity_level>=2: 
                f_out.write("new value of k_in_grid is: %i\n" %(k_in_grid))
                f_out.flush()
            # calculate real value of predicted coordinates
            for j in range(param):
                x_real.append(dim_list[j][k_in_grid])
            y_real=G_list[k_in_grid]
            # Calculate ML prediction relative error and MLgain
            error_ML = abs(( min(y2b) - y_real) / y_real)
            #MLgain_pred = ( min(y2a) - min(y2b))/abs(min(y2a))
            #MLgain_real = ( min(y2a) - y_real)/abs(min(y2a))
            # calculate MLgain with respect to minimum value obtained with a standard exploration, whether in t1 or t2
            #MLgain_pred = ( min_standard - min(y2b))/(min(y1)-min_G)
            #MLgain_real = ( min_standard -  y_real )/(min(y1)-min_G)
            if min(y2a) < min(y1):
                min_standard = min(y2a)
            else:
                min_standard = min(y1)
            MLgain_pred = min_standard - min(y2b)
            MLgain_real = min_standard -  y_real 
            MLgain_real_relative = (min_standard -  y_real )/abs(min_standard)
            # Print t2 exploration results
            if verbosity_level>=1: 
                f_out.write("################ \n")
                f_out.write("## Initial random exploration: \n")
                f_out.write("t0 = %i \n" %(t0_time))
                f_out.write("################ \n")
                f_out.write("################ \n")
                f_out.write("## t1 exploration: \n")
                f_out.write("t1 = %i \n" %(t1_time))
                f_out.write("Last value: X1 index (unique timestep): %i\n" %(len(y1)-1))
                f_out.write("Last value: X1: %s, y: %s\n" %(str(X1[-1]),str(y1[-1])))
                f_out.write("Minimum value: X1 index (unique timestep): %s\n" %(str(np.where(y1 == np.min(y1))[0][0])))
                f_out.write("Minimum value: X1: %s, y: %s\n" %(str(X1[np.where(y1 == np.min(y1))][0]),str(min(y1))))
                f_out.write("################ \n")
                f_out.write("################ \n")
                f_out.write("## t2 standard exploration: \n")
                f_out.write("t2 = %i \n" %(t2_time))
                f_out.write("Last value: X2a index (unique timestep): %i\n" %(len(y2a)-1))
                f_out.write("Last value: X2a: %s, y2a: %s\n" %(str(X2a[-1]),str(y2a[-1])))
                f_out.write("Minimum value: X2a index (unique timestep): %s\n" %(str(np.where(y2a == np.min(y2a))[0][0])))
                f_out.write("Minimum value: X2a: %s, y2a: %s\n" %(str(X2a[np.where(y2a == np.min(y2a))][0]),str(min(y2a))))
                f_out.write("################ \n")
                f_out.write("################ \n")
                f_out.write("## t2 ML exploration: \n")
                f_out.write("t2 = %i \n" %(t2_time))
                f_out.write("Last value: X2b index (unique timestep): %i\n" %(len(y2b)-1))
                f_out.write("Last value: X2b: %s, y2b: %s\n" %(str(X2b[-1]),str(y2b[-1])))
                f_out.write("Minimum value: X2b index (unique timestep): %s\n" %(str(np.where(y2b == np.min(y2b))[0][0])))
                f_out.write("Minimum predicted value: X2b: %s, y2b: %s\n" %(str(X2b[np.where(y2b == np.min(y2b))][0]),str(min(y2b))))
                f_out.write("Minimum real value: X2b: %s, y2b: %s\n" %(str(x_real),str(y_real)))
    
                f_out.write("################ \n")
                f_out.write("################ \n")
                f_out.write("## ML benefits: \n")
                f_out.write("Predicted MLgain: %.3f \n" %(MLgain_pred))
                f_out.write("Real MLgain: %.3f \n" %(MLgain_real))
                f_out.write("Real relative MLgain: %.3f \n" %(MLgain_real_relative))
                f_out.write("ML prediction relative error: %.3f \n" %(error_ML))
                f_out.write("################ \n")
                f_out.flush()

            ML_benefits=[]
            ML_benefits.append(MLgain_pred)
            ML_benefits.append(MLgain_real)
            ML_benefits.append(error_ML)
            ML_benefits.append(min_standard)
            ML_benefits.append(y_real)
            ML_benefits.append(MLgain_real_relative)
            if verbosity_level>=1: 
                f_out.write("For each Nwalker: %s\n" %(str(ML_benefits)))
                f_out.flush()
            ML_benefits_list.append(ML_benefits)
            result2=ML_benefits_list
    time_taken2 = time()-start
    if verbosity_level>=1:
        f_out.write("Rest of MLL took %0.4f seconds\n" %(time_taken2-time_taken1))
        f_out.write("I am returning these values: %s, %s\n" %(str(result1), str(result2)))
        f_out.close()
    return (result1, result2)

def generate_grid(iseed,l):
    initial_seed = iseed
    time_taken1 = time()-start
    Amplitude      = []
    center_N       = [[] for i in range(N)]
    width_N        = [[] for i in range(N)]
    dim_list       = [[] for i in range(param)]
    G_list         = []
    f_out = open('%s_%s.log' % (grid_name,l), 'w')
    if verbosity_level>=1: 
        f_out.write('## Start: "generate_grid" function \n')
        f_out.write("########################### \n")
        f_out.write("###### Landscape %i ####### \n" % (l))
        f_out.write("########################### \n")
        f_out.write("%s %i %s %6.2f \n" % ('Initial seed:', iseed, '. Verbosity level:', verbosity_level))
        f_out.flush()
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
    if verbosity_level>=2:
        #f_out.write("%4s %14s %22s %34s \n" % ("N","Amplitude","Center","Width"))
        for i in range(len(Amplitude)):
            line1 = []
            line2 = []
            for j in range(param):
                line1.append((center_N[i][j]))
                line2.append((width_N[i][j]))
            #f_out.write("%4i %2s %10.6f %2s %s %2s %s \n" % (i, "", Amplitude[i],"",str(line1),"",str(line2)))
        f_out.flush()
    # CALCULATE G GRID #
    counter=0
    if verbosity_level>=1: f_out.write("%8s %11s %15s \n" % ("i","x","G"))
    Nlen=(round(int((grid_max-grid_min)/(grid_Delta)+1)))
    for index_i in itertools.product(range(Nlen), repeat=param):
        for j in range(param):
            dim_list[j].append(index_i[j]*grid_Delta)
        G=0.0
        for i in range(N):
            gauss=0.0
            for dim in range(param):
                gauss=gauss+((dim_list[dim][counter]-center_N[i][dim])**2/(2.0*width_N[i][dim]**2))
            G = G + Amplitude[i] * math.exp(-gauss)
        G_list.append(G)
        line = []
        for j in range(param):
            line.append((dim_list[j][counter]))
        line.append(G_list[counter])
        if verbosity_level>=1: 
            f_out.write("%8i  " %(counter))
            for i in range(param+1):
                f_out.write("%f   " %(line[i]))
            f_out.write("\n")
            f_out.flush()
        counter=counter+1

    Ngrid=int((grid_max/grid_Delta+1)**param)   # calculate number of grid points
    max_G=max(G_list)
    min_G=min(G_list)
    if verbosity_level>=1:
        max_G_index=int(np.where(G_list == np.max(G_list))[0])
        min_G_index=int(np.where(G_list == np.min(G_list))[0])
        f_out.write("Number of grid points: %i \n" %Ngrid)
        line1 = []
        line2 = []
        for j in range(param):
            line1.append(dim_list[j][max_G_index])
            line2.append(dim_list[j][min_G_index])
        line1.append(max_G)
        line2.append(min_G)
        f_out.write("Maximum value of grid: %s \n" %(str(line1)))
        f_out.write("Minimum value of grid: %s \n" %(str(line2)))
        f_out.flush()

    if plot_contour_map == True and param == 2:
        plot('contour',l,None,initial_seed,dim_list,G_list,None,None,None,None,None)
    #return dim_list, G_list, Ngrid, max_G
    time_taken2 = time()-start
    f_out.write("Generate grid took %0.4f seconds\n" %(time_taken2-time_taken1))
    return None

def check_input_values():
    if type(dask_parallel) != bool:
        print ('INPUT ERROR: dask_parallel should be boolean, but is:', dask_parallel)
        sys.exit()
    if type(t1_analysis) != bool:
        print ('INPUT ERROR: t1_analysis should be boolean, but is:', t1_analysis)
    if type(t2_exploration) != bool:
        print ('INPUT ERROR: t2_exploration should be boolean, but is:', t2_exploration)
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
    if verbosity_level not in allowed_verbosity_level:
        print ('INPUT ERROR: verbosity_level needs to be in',allowed_verbosity_level, ', but is:', verbosity_level)
    if t2_ML not in allowed_t2_ML:
        print ('INPUT ERROR: ML needs to be in',allowed_t2_ML, ', but is:', t2_ML)
        sys.exit()
        sys.exit()
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print("\n")
    print('##### START PRINT INPUT  #####')
    print('##############################')
    print('# General Landscape parameters')
    print('##############################')
    print('### Parallel computing ###')
    print('dask_parallel =',dask_parallel)
    print('NCPU',NCPU)
    print('### Verbose ###')
    print('verbosity_level =',verbosity_level)
    print('allowed_verbosity_level =',allowed_verbosity_level)
    print('log_name',log_name)
    print('### Landscape parameters ###')
    print('Nspf =',Nspf)
    print('initial_spf =',initial_spf)
    print('S =',S)
    print('iseed =',iseed)
    print('param =',param)
    print('center_min =',center_min)
    print('center_max =',center_max)
    print('### Grid parameters ###')
    print('grid_min =',grid_min)
    print('grid_max =',grid_max)
    print('grid_Delta =',grid_Delta)
    print('calculate_grid =',calculate_grid)
    print('grid_name =',grid_name)
    print('plot_t1_exploration =',plot_t1_exploration)
    print('plot_t1_error_metric =',plot_t1_error_metric)
    print('plot_contour_map =',plot_contour_map)
    print('##############################')
    print('# T1 exploration parameters')
    print('##############################')
    print('Nwalkers =',Nwalkers)
    print('adven =',adven)
    print('t0_time =',t0_time)
    print('t1_time =',t1_time)
    print('d_threshold =',d_threshold)
    print('initial_sampling =',initial_sampling)
    print('allowed_initial_sampling =',allowed_initial_sampling)
    print('##############################')
    print('# T2 exploration parameters')
    print('##############################')
    print('t2_exploration =',t2_exploration)
    print('t2_time =',t2_time)
    print('t2_train_time =',t2_train_time)
    print('t2_ML =',t2_ML)
    print('allowed_t2_ML =',allowed_t2_ML)
    print('##############################')
    print('# Error metric parameters')
    print('##############################')
    print('t1_analysis =',t1_analysis)
    print('error_metric =',error_metric)
    print('allowed_error_metric =',allowed_error_metric)
    print('ML =',ML)
    print('allowed_ML =',allowed_ML)
    print('CV =',CV)
    print('allowed_CV =',allowed_CV)
    print('k_fold =',k_fold)
    print('test_last_percentage =',test_last_percentage)
    if ML=='kNN':
        print('### kNN parameters ###')
        print('n_neighbor =',n_neighbor)
        print('weights =',weights)
    if ML=='GBR':
        print('### GBR parameters ###')
        print('GBR_criterion =',GBR_criterion)
        print('GBR_n_estimators =',GBR_n_estimators)
        print('GBR_learning_rate =',GBR_learning_rate)
        print('GBR_max_depth =',GBR_max_depth)
        print('GBR_min_samples_split =',GBR_min_samples_split)
        print('GBR_min_samples_leaf =',GBR_min_samples_leaf)
    if ML=='GPR':
        print('### GPR parameters ###')
        print('GPR_alpha =',GPR_alpha)
        print('GPR_length_scale =',GPR_length_scale)
        print('GPR_alpha_lim =',GPR_alpha_lim)
        print('GPR_length_scale_lim =',GPR_length_scale_lim)
    if ML=='KRR':
        print('### KRR parameters ###')
        print('KRR_alpha =',KRR_alpha)
        print('KRR_kernel =',KRR_kernel)
        print('KRR_gamma =',KRR_gamma)
        print('optimize_KRR_hyperparams =',optimize_KRR_hyperparams)
        print('KRR_alpha_lim =',KRR_alpha_lim)
        print('KRR_gamma_lim =',KRR_gamma_lim)
    print('##############################')
    print('# Differential evolution parameters')
    print('diff_popsize =', diff_popsize)
    print('diff_tol =', diff_tol)
    print('##############################')
    print('### Calculated parameters ###')
    print('width_min =',width_min)
    print('width_max =',width_max)
    print('Amplitude_min =',Amplitude_min)
    print('Amplitude_max =',Amplitude_max)
    print('N =',N)
    print('#####   END PRINT INPUT  #####')
    print("\n",flush=True)

#def explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,t0,t1,t2,Xi,yi,ML_explore,X0,y0):
def explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,t0,t1,t2,Xi,yi,ML_explore,X0,y0):
    walker_x       = []
    path_x         = [[] for i in range(param)]
    path_G         = []
    list_t         = []
    prob           = []
    neighbor_walker= [[] for i in range(param)]
    neighbor_G     = []
    x_param        = [[] for j in range(param)]
    y              = []
    P              = int(round(d_threshold/grid_Delta + 1))
    Nx             = ((grid_max-grid_min)/grid_Delta)+1
    # print header
    if verbosity_level>=1: 
        f_out.write('## Start: "explore_landscape" function \n')
        f_out.write("############# \n")
        f_out.write("Start explorer %i \n" % (w))
        f_out.write("Adventurousness: %f \n" % (adven[w]))
        f_out.write("############# \n")
        f_out.write("Number of points per dimension: %i \n" %Nx)
        f_out.write("Testing w: %i, iseed: %i \n" % (w,iseed))
        f_out.flush()
    ### Perform t0 exploration ###
    if verbosity_level>=1 and t0 !=0: 
        f_out.write("## Start random biased exploration\n" %())
        f_out.flush()
    if t0 != 0:
        #print('1-TEST inside function, X0:')
        #print(X0)
        for t in range(t0):
            for i in range(param):
                if initial_sampling=='different': iseed=iseed+w+l+i+t
                if initial_sampling=='same': iseed=iseed+1
                random.seed(iseed)
                num=int(random.randint(0,Ngrid-1))
                if t==0:
                    walker_x.append(dim_list[i][num])
                else:
                    walker_x[i]=dim_list[i][num]
                #print('TEST:', t,i,num,dim_list[i][num],walker_x[i])
            for i in range(param):
                path_x[i].append(walker_x[i])
            num_in_grid=0.0
            for i in range(param):
                num_in_grid=num_in_grid + walker_x[param-1-i]*(Nx**i)*(Nx-1)
            num_in_grid=int(round(num_in_grid))
            path_G.append(G_list[num_in_grid])
            list_t.append(t)
    
            if verbosity_level>=1:
                line = []
                for j in range(param):
                    line.append((walker_x[j]))
                line.append((G_list[num_in_grid]))
                f_out.write("timestep %4i %2s %s \n" %(t,"",str(line)))
                f_out.flush()
        #print('TEST INITIAL POINTS:')
        #print(path_x)
        #print(path_G)
        if t1 == 0:
            X = path_x
            y = path_G
            return X,y,len(y)
        #print('2-TEST inside function, X0:')
        #print(X0)
    # Set values for t1 exploration
    #print('3-TEST inside function, X0:')
    #print(X0)
    if t2==0:
        #print('TEST INITIAL POINTS:')
        #print(X0)
        #print(y0)
        #path_x = X0
        #path_G = y0
        #path_x = list(X0)
        #path_G = list(y0)
        path_x = copy.deepcopy(X0)
        path_G = copy.deepcopy(y0)
        #path_x = X0[:]
        #path_G = y0[:]
        #path_x = X0.copy()
        #path_G = y0.copy()
        #path_x = []; path_x.extend(X0)
        #path_G = []; path_G.extend(y0)
        #print('ID 1:', id(path_x))
        #print('ID 2:', id(X0))
        #t_ini=t0
        #t_fin=t1+t0
        t_ini=t0_time
        t_fin=t1+t0_time
        #print('TEST INITIAL POINTS:')
        #print(path_x)
        #print(path_G)
        for i in range(param): # set walker_x to last path_x
            #walker_x[i]=path_x[i][-1]
            walker_x.append(path_x[i][-1])
        #print('4-TEST inside function, X0:')
        #print(X0)
    # Set values for t2 exploration
    elif t0==0 and t2!=0:
        t_ini=0
        t_fin=t2
        for t in range(t1): # copy Xi and yi to path_x and path_G
            for i in range(param):
                path_x[i].append(Xi[t][i])
            path_G.append(yi[t])
        for i in range(param): # set walker_x to last path_x
            walker_x.append(Xi[-1][i])
    ### Perform t1 and t2 standard exploration ###
    if ML_explore == False:
        #print('5-TEST inside function, X0:')
        #print(X0)
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
            # Verbosity
            if verbosity_level>=2:
                f_out.write("Special points: %i \n" % (special_points))
                for i in range(special_points):
                    line = []
                    for j in range(param):
                        line.append((minimum_path_x[j][i]))
                    line.append((minimum_path_G[i]))
                    f_out.write("%i %s \n" % (i,str(line)))
                line = []
                for j in range(param):
                    line.append((minimum_path_x[j][draw]))
                line.append((minimum_path_G[draw]))
                f_out.write("Selected point in draw: %i %s \n" % (i,str(line)))
                f_out.write("draw %s, draw_in_grid %s,draw_in_grid_list %s \n" % (str(draw),str(draw_in_grid),str(draw_in_grid_list)))
                f_out.write("Consider nearby points: \n")
                f_out.write("%6s %11s %19s %12s %12s \n" % ("i","x","G","Prob","distance"))
                f_out.flush()
            #print('FINAL TEST dim_list', dim_list)
            #print('FINAL TEST G_list', G_list)
            #print('FINAL TEST minimum_path_x', minimum_path_x)
            #print('FINAL TEST minimum_path_G', minimum_path_G)
            # Check for inconsistencies
            for k in range(len(draw_in_grid_list)):
                counter_param=0
                for i in range(param):
                    if minimum_path_x[i][draw] == dim_list[i][draw_in_grid_list[k]]:
                        counter_param=counter_param+1
                if counter_param==param:
                    #print('I am in point #:', k)
                    draw_in_grid=draw_in_grid_list[k]
                    break
            # initialize prob and neighbor_XX values
            for i in range((P*2+1)**param):
                    prob.append(0.0)
                    neighbor_G.append(0.0)
                    for j in range(param):
                        neighbor_walker[j].append(0.0)
            counter3=0
            # Index_i serves to explore neighbors: from -P+1 to P for each parameter
            for index_i in itertools.product(range(-P+1,P), repeat=param):
                try: # Use try/except to ignore errors when looking for points outside of landscape
                    index=[]
                    subs=0.0
                    # Assign indeces of grid corresponding to wanted points
                    for j in range(param):
                        index.append(draw_in_grid - Nx**(param-1-j)*index_i[j])
                        subs=subs+Nx**(param-1-j)*index_i[j]
                        index[j]=int(index[j])
                    index.append(draw_in_grid-subs)
                    index[param]=int(index[param])
                    # Calculate distance between point of interest and its neighbors
                    dummy_dist = 0.0
                    for j in range(param):
                        dummy_dist = dummy_dist + (minimum_path_x[j][draw]-dim_list[j][index[j]])**2
                    d_ij = (math.sqrt(dummy_dist))
                    # If distance is within threshold: consider that point
                    if d_ij <= d_threshold and d_ij > 0.0:
                        use_this_point=True
                        # Also check that does not correspond to a point explored previously
                        for i in range(len(path_G)):
                            counter_false=0
                            for j in range(param):
                                if dim_list[j][index[j]] == path_x[j][i]: counter_false=counter_false+1
                            if counter_false==param: use_this_point=False
                        # If those two conditions are fulfilled, update prob[counter3] and prob_sum
                        if use_this_point==True:
                            for j in range(param):
                                neighbor_walker[j][counter3]=dim_list[j][index[j]]
                            neighbor_G[counter3]=G_list[index[param]]
                            prob[counter3]=1.0
                            prob_sum=prob_sum+prob[counter3]
                    if verbosity_level>=2:
                        line = []
                        for j in range(param):
                            line.append((dim_list[j][index[j]]))
                        line.append((G_list[index[param]]))
                        f_out.write("%6i %s %2s %5.1f %2s %10.6f \n" % (counter3,line,"",prob[counter3],"",d_ij))
                        f_out.flush()
                except:
                    pass
                counter3=counter3+1
            # Check for inconsistencies
            if prob_sum==0.0: 
                print("STOP - ERROR: No new candidate points found within threshold",flush=True)
                print("STOP - ERROR: At Nspf:", l, ". Walker:", w, ". Time:", t,flush=True)
                sys.exit()

            if len(range((P*2+1)**param)) != len(prob):
                print("STOP - ERROR: Problem with number of nearby points considered for next step",flush=True)
                sys.exit()
            if verbosity_level>=2: 
                f_out.write("Number of points considered: %i \n" % (len(range(((P-1)*2+1)**param))))
                f_out.write("Points within threshold: %f \n" % int(round((prob_sum))))
                f_out.flush()
            # renormalize probabilities
            for i in range(counter3):
                prob[i]=prob[i]/prob_sum
            # choose neighbor to draw
            draw=int(choice(range((P*2+1)**param),size=1,p=prob))
            # add draw's X and G values to path_x and path_G
            #print('5a-TEST inside function, X0:')
            #print(X0)
            for i in range(param):
                walker_x[i]=neighbor_walker[i][draw]
                path_x[i].append(walker_x[i])
            path_G.append(neighbor_G[draw])
            list_t.append(t)
            #print('5b-TEST inside function, X0:')
            #print(X0)
            # verbosity
            if verbosity_level>=2: 
                f_out.write("We draw neighbor no.: %6i\n" % (draw))
                f_out.flush()
            if verbosity_level>=1:
                line = []
                for j in range(param):
                    line.append(walker_x[j])
                line.append(neighbor_G[draw])
                f_out.write("timestep %6i %2s %s\n" % (t,"",str(line)))
                f_out.flush()
            # update x_param and y with new values
            for i in range(param):
                x_param[i].append(walker_x[i])
            y.append(neighbor_G[draw])
        # calculate final X and y
        X,y = create_X_and_y(f_out,x_param,y)
        #print('6-TEST inside function, X0:')
        #print(X0)
    ### Perform t2 ML exploration
    else:
        for t in range(t_ini,t_fin):
            time_taken0 = time()-start
            time_taken1 = time()-start
            if verbosity_level>=1: 
                f_out.write("## Start ML Exploration\n" %())
                f_out.flush()
            #time_taken2 = time()-start
            #f_out.write("time: %i. Test1 took %0.4f seconds \n" %(t,time_taken2-time_taken1))
            # For t2=0, calculate bubble for all points previously visited (slow process)
            if t==0:
                # initialize values
                x_bubble=[[] for j in range(param)]
                y_bubble=[]
                # For each point in Xi
                for k in range(len(path_G)):
                    time_taken1 = time()-start
                    counter3=0
                    del prob[:]
                    del neighbor_walker[:][:]
                    del neighbor_G[:]
                    #prob = []
                    #neighbor_walker = [[] for i in range(param)]
                    #neighbor_G = []
                    prob_sum=0.0
                    # get coordinates of kth point in SPF grid
                    new_k_in_grid=[[] for j in range(param)]
                    # for each parameter
                    k_list=[]
                    #f_out.write("croqueta test: looking for point %f %f %f\n" %(path_x[0][k],path_x[1][k],path_x[2][k]))
                    for j in range(param):
                        # calculate list of indeces in grid that match path_x values
                        k_list = [i for i, x in enumerate(dim_list[j]) if x == path_x[j][k]]
                        new_k_in_grid[j].append(k_list)
                        if verbosity_level>=2: f_out.write("Looking for path_x[j][k]: %f\n" %(path_x[j][k]))
                        #for l in range(len(k_list)):
                             #f_out.write("new croqueta test: l %i, dim_list[j][l] %f\n" %(l, dim_list[j][l]))
                        #f_out.write("croqueta test k_list: %s\n" %(str(k_list)))
                        #f_out.write("croqueta test len(k_list): %i\n" %(len(k_list)))
                        #f_out.write("croqueta test new_k_in_grid: %s \n" %(str(new_k_in_grid)))
                    for i in range(len(k_list)):
                        counter_k=0
                        for j in range(1,param):
                            #f_out.write("croqueta test k %i, i %i, j %i\n" %(k,i,j))
                            #f_out.flush()
                            #f_out.write("croqueta test new_k_in_grid[0][0][i]: %s\n" %( str(new_k_in_grid[0][0][i])))
                            #f_out.flush()
                            #f_out.write("croqueta test new_k_in_grid[j][0]: %s\n" %( str(new_k_in_grid[j][0])))
                            #f_out.flush()
                            if new_k_in_grid[0][0][i] in new_k_in_grid[j][0]:
                                counter_k=counter_k+1
                                if counter_k==param-1:
                                    k_in_grid=new_k_in_grid[0][0][i]
                    if verbosity_level>=2: 
                        f_out.write("value of k_in_grid is: %i\n" %(k_in_grid))
                        f_out.flush()
                    #time_taken2 = time()-start
                    #f_out.write("time: %i. Test2 took %0.4f seconds \n" %(t,time_taken2-time_taken1))
                    #######################
                    time_taken1 = time()-start
                    # initialize neighbor_G and neighbor_walker[j]
                    for i in range((P*2+1)**param):
                        prob.append(0.0)
                        neighbor_G.append(0.0)
                        for j in range(param):
                            neighbor_walker[j].append(0.0)
                    # Check points within threshold
                    ####### for any param #######
                    if verbosity_level>=2:
                    #if verbosity_level>=1:
                        line = []
                        for j in range(param):
                            line.append((path_x[j][k]))
                        line.append((path_G[k]))
                        f_out.write("Check around point: %s\n" %(str(line)))
                        f_out.write("%6s %20s %11s %13s \n" % ("i","[x, G]","Prob","distance"))
                        f_out.flush()
                    time_taken2 = time()-start
                    #f_out.write("time: %i. Test3 took %0.4f seconds \n" %(t,time_taken2-time_taken1))
                    ## Index_i serves to explore neighbors: from -P+1 to P for each parameter
                    for index_i in itertools.product(range(-P+1,P), repeat=param):
                        time_taken1 = time()-start
                        try: # Use try/except to ignore errors when looking for points outside of landscape
                            index=[]
                            subs=0.0
                            # Assign indeces of grid corresponding to wanted points
                            for j in range(param):
                                index.append(k_in_grid - Nx**(param-1-j)*index_i[j])
                                subs=subs+Nx**(param-1-j)*index_i[j]
                                index[j]=int(index[j])
                            index.append(k_in_grid-subs)
                            index[param]=int(index[param])
                            # Calculate distance between point of interest and its neighbors
                            dummy_dist = 0.0
                            for j in range(param):
                                dummy_dist = dummy_dist + (path_x[j][k]-dim_list[j][index[j]])**2
                            d_ij = (math.sqrt(dummy_dist))
                            # If distance is within threshold: consider that point
                            if d_ij <= d_threshold and d_ij > 0.0:
                                use_this_point=True
                                # Also check that does not correspond to a point explored previously
                                for i in range(len(path_G)):
                                    counter_false=0
                                    for j in range(param):
                                        if dim_list[0][index[j]] == path_x[j][i]: counter_false=counter_false+1
                                    if counter_false==param: use_this_point=False
                                # If those two conditions are fulfilled, update prob[counter3] and prob_sum
                                if use_this_point==True:
                                    for j in range(param):
                                        neighbor_walker[j][counter3]=dim_list[j][index[j]]
                                    neighbor_G[counter3]=G_list[index[param]]
                                    prob[counter3]=1.0
                                    prob_sum=prob_sum+prob[counter3]
                                # Add all valid neighbors to 'bubble'
                                for j in range(param):
                                    x_bubble[j].append(neighbor_walker[j][counter3])
                                y_bubble.append(neighbor_G[counter3])
    
                            if verbosity_level>=2:
                            #if verbosity_level>=1:
                                line = []
                                for j in range(param):
                                    line.append((dim_list[j][index[j]]))
                                line.append((G_list[index[param]]))
                                f_out.write("%6i %s %2s %5.1f %2s %10.6f \n" % (counter3,line,"",prob[counter3],"",d_ij))
                                f_out.flush()
                        except:
                            pass
                        counter3=counter3+1
            # For t>0, calculate just bubble around new point, and add it to previous bubble (fast)
            elif t>0:
                # For min point calculated last step:
                if verbosity_level>=1:
                    f_out.write("I am in t: %i, and last added point was: %s\n" %(t,str(min_point)))
                time_taken1 = time()-start
                counter3=0
                del prob[:]
                del neighbor_walker[:][:]
                del neighbor_G[:]
                prob_sum=0.0
                # get coordinates of kth point in SPF grid
                new_k_in_grid=[[] for j in range(param)]
                for j in range(param):
                    k_list = [i for i, x in enumerate(dim_list[j]) if x == min_point[j]]
                    new_k_in_grid[j].append(k_list)
                for i in range(len(k_list)):
                    counter_k=0
                    for j in range(1,param):
                        if new_k_in_grid[0][0][i] in new_k_in_grid[j][0]:
                            counter_k=counter_k+1
                            if counter_k==param-1:
                                k_in_grid=new_k_in_grid[0][0][i]
                #if verbosity_level>=2: 
                if verbosity_level>=1: 
                    f_out.write("value of k_in_grid is: %i\n" %(k_in_grid))
                    f_out.flush()
                #time_taken2 = time()-start
                #f_out.write("time: %i. Test2 took %0.4f seconds \n" %(t,time_taken2-time_taken1))
                #######################
                time_taken1 = time()-start
                # initialize neighbor_G and neighbor_walker[j]
                for i in range((P*2+1)**param):
                    prob.append(0.0)
                    neighbor_G.append(0.0)
                    for j in range(param):
                        neighbor_walker[j].append(0.0)
                # Check points within threshold
                ####### for any param #######
                if verbosity_level>=2:
                    line = []
                    for j in range(param):
                        line.append((dim_list[j][k_in_grid]))
                    line.append((G_list[k_in_grid]))
                    f_out.write("New Check around point: %s\n" %(str(line)))
                    f_out.write("%6s %20s %11s %13s \n" % ("i","[x, G]","Prob","distance"))
                    f_out.flush()
                time_taken2 = time()-start
                #f_out.write("time: %i. Test3 took %0.4f seconds \n" %(t,time_taken2-time_taken1))
                ## Index_i serves to explore neighbors: from -P+1 to P for each parameter
                for index_i in itertools.product(range(-P+1,P), repeat=param):
                    time_taken1 = time()-start
                    try: # Use try/except to ignore errors when looking for points outside of landscape
                        index=[]
                        subs=0.0
                        # Assign indeces of grid corresponding to wanted points
                        for j in range(param):
                            index.append(k_in_grid - Nx**(param-1-j)*index_i[j])
                            subs=subs+Nx**(param-1-j)*index_i[j]
                            index[j]=int(index[j])
                        index.append(k_in_grid-subs)
                        index[param]=int(index[param])
                        #f_out.write("test index: %s\n" %(str(index)))
                        # Calculate distance between point of interest and its neighbors
                        dummy_dist = 0.0
                        for j in range(param):
                            dummy_dist = dummy_dist + (dim_list[j][k_in_grid]-dim_list[j][index[j]])**2
                        d_ij = (math.sqrt(dummy_dist))
                        # If distance is within threshold: consider that point
                        if d_ij <= d_threshold and d_ij > 0.0:
                            use_this_point=True
                            # Also check that does not correspond to a point explored previously
                            for i in range(len(path_G)):
                                counter_false=0
                                for j in range(param):
                                    if dim_list[0][index[j]] == path_x[j][i]: counter_false=counter_false+1
                                if counter_false==param: use_this_point=False
                            # If those two conditions are fulfilled, update prob[counter3] and prob_sum
                            if use_this_point==True:
                                for j in range(param):
                                    neighbor_walker[j][counter3]=dim_list[j][index[j]]
                                neighbor_G[counter3]=G_list[index[param]]
                                prob[counter3]=1.0
                                prob_sum=prob_sum+prob[counter3]
                            # Add all valid neighbors to 'bubble'
                            for j in range(param):
                                x_bubble[j].append(neighbor_walker[j][counter3])
                            y_bubble.append(neighbor_G[counter3])
                        if verbosity_level>=2:
                            line = []
                            for j in range(param):
                                line.append((dim_list[j][index[j]]))
                            line.append((G_list[index[param]]))
                            f_out.write("%6i %s %2s %5.1f %2s %10.6f \n" % (counter3,line,"",prob[counter3],"",d_ij))
                            f_out.flush()
                    except:
                        pass
                    counter3=counter3+1
            if verbosity_level>=1:
                f_out.write("Number of points in bubble before removing duplicates: %i\n" %(len(y_bubble)))
            if verbosity_level>=2:
                for j in range(param):
                    f_out.write("x_bubble[%i]: %s\n" %(j,str(x_bubble[j])))
                f_out.flush()
            # train ML with path_x and path_G (all points explored previously)
            # predict min_point with ML for x_bubble and y_bubble (all neighbour points not explored previously)
            path_x,path_G = create_X_and_y(f_out,path_x,path_G)
            x_bubble,y_bubble = create_X_and_y(f_out,x_bubble,y_bubble)
            time_taken2 = time()-start
            if verbosity_level>=1:
                f_out.write("Number of points in bubble after removing duplicates: %i\n" %(len(y_bubble)))
                f_out.write("Explore landscape ML. time: %i. Before ML took %0.4f seconds \n" %(t,time_taken2-time_taken0))
            if t2_ML=='kNN': min_point=kNN(x_bubble,y_bubble,iseed,l,w,f_out,path_x,path_G,2,t)
            #if t2_ML=='GPR': min_point=GPR(x_bubble,y_bubble,iseed,l,w,f_out,path_x,path_G,2,t)
            if t2_ML=='GPR':
                hyperparams=[GPR_alpha,GPR_length_scale]
                if optimize_GPR_hyperparams == False:
                    min_point=GPR(hyperparams,x_bubble,y_bubble,iseed,l,w,f_out,path_x,path_G,2,t)
                else:
                    mini_args=(path_x,path_G,iseed,l,w,f_out,None,None,1,t) # get rmse fitting previous points
                    bounds = [GPR_alpha_lim]+[GPR_length_scale_lim]
                    solver=differential_evolution(GPR,bounds,args=mini_args,popsize=diff_popsize,tol=diff_tol)
                    best_hyperparams = solver.x
                    best_rmse = solver.fun
                    if verbosity_level>=1: 
                        f_out.write("Best hyperparameters: %s \n" %str(best_hyperparams))
                        f_out.write("Best rmse: %f \n" %best_rmse)
                        f_out.flush()
                    hyperparams=[best_hyperparams[0],best_hyperparams[1]]
                    min_point=GPR(hyperparams,x_bubble,y_bubble,iseed,l,w,f_out,path_x,path_G,2,t)
            if t2_ML=='KRR':
                hyperparams=[KRR_alpha,KRR_gamma]
                if optimize_KRR_hyperparams == False:
                    min_point=KRR(hyperparams,x_bubble,y_bubble,iseed,l,w,f_out,path_x,path_G,2,t)
                else:
                    mini_args=(path_x,path_G,iseed,l,w,f_out,None,None,1,t) # get rmse fitting previous points
                    bounds = [KRR_alpha_lim]+[KRR_gamma_lim]
                    solver=differential_evolution(KRR,bounds,args=mini_args,popsize=diff_popsize,tol=diff_tol)
                    best_hyperparams = solver.x
                    best_rmse = solver.fun
                    if verbosity_level>=1: 
                        f_out.write("Best hyperparameters: %s \n" %str(best_hyperparams))
                        f_out.write("Best rmse: %f \n" %best_rmse)
                        f_out.flush()
                    hyperparams=[best_hyperparams[0],best_hyperparams[1]]
                    min_point=KRR(hyperparams,x_bubble,y_bubble,iseed,l,w,f_out,path_x,path_G,2,t)
            #time_taken1 = time()-start
            # print x_bubble corresponding to min(y_bubble)
            if verbosity_level>=1: 
                f_out.write("## At time %i, minimum predicted point is: %s\n" %(t, str(min_point)))
                f_out.flush()
            # prepare for next timestep
            path_x=path_x.tolist()
            path_G=path_G.tolist()
            x_bubble=x_bubble.tolist()
            y_bubble=y_bubble.tolist()
            ####### croqueta ########
            #f_out.write("croqueta param: %s\n" %(str(param)))
            #f_out.write("croqueta type(param): %s\n" %(type(param)))
            #f_out.write("min_point: %s\n" %(min_point))
            #f_out.write("min_point[0:param]\n" %(min_point[0:param]))
            #f_out.write("min_point[param]\n" %(min_point[param]))
            #########################
            path_x.append(min_point[0:param])
            path_G.append(min_point[param])
            path_x=[list(i) for i in zip(*path_x)]
            x_bubble=[list(i) for i in zip(*x_bubble)]
            # add coordinates to final array
            for i in range(param):
                x_param[i].append(min_point[i])
            y.append(min_point[param])
            #time_taken2 = time()-start
            #f_out.write("Explore landscape ML. time: %i. After ML took %0.4f seconds \n" %(t,time_taken2-time_taken1))
        # get final X,y
        X,y = create_X_and_y(f_out,x_param,y)
    return X,y,len(y)

# CREATE X AND y #
def create_X_and_y(f_out,x_param,y):
    zippedList = list(zip(*x_param,y))
    df=pd.DataFrame(zippedList)
    df2=df.drop_duplicates(subset=param, keep="first")
    #df2=df
    df_X=df2[range(0,param)]
    df_y=df2[param]
    X=df_X.to_numpy()
    y=df_y.to_numpy()

    if verbosity_level>=1: 
        f_out.write("## Unique points: %i\n" % (len(y)))
        f_out.write("## X: \n")
        f_out.write("%s \n" % (str(X)))
        f_out.write("## y: \n")
        f_out.write("%s \n" % (str(y)))
        f_out.flush()
    return X,y

# CALCULATE k-NN #
def kNN(X,y,iseed,l,w,f_out,Xtr,ytr,mode,t):
    # initialize values
    iseed=iseed+1
    average_r=0.0
    average_r_pearson=0.0
    average_rmse=0.0
    real_y=[]
    predicted_y=[]
    counter_split=0
    prev_total_rmse = 0
    # CASE1: Calculate error metric
    if mode==1:
        for n in range(len(n_neighbor)):
            # verbose info
            if verbosity_level>=1: 
                f_out.write('## Start: "kNN" function \n')
                f_out.write('-------- \n')
                f_out.write('Perform k-NN \n')
                f_out.write('k= %i \n' % (n_neighbor[n]))
                f_out.write('cross_validation %i - fold \n' % (k_fold))
                f_out.write('weights %s \n' % (weights))
                f_out.write('iseed %s \n' % (iseed))
                f_out.write('-------- \n')
                f_out.flush()
            # assign splits to kf or and loo
            if CV=='kf':
                kf = KFold(n_splits=k_fold,shuffle=True,random_state=iseed)
                validation=kf.split(X)
            if CV=='loo':
                loo = LeaveOneOut()
                validation=loo.split(X)
    
            # For kf and loo
            if CV=='kf' or CV=='loo':
                # calculate r and rmse for each split
                for train_index, test_index in validation:
                    # assign train and test data
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    # scale data
                    scaler = preprocessing.StandardScaler().fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    # fit kNN with (X_train_scaled, y_train) and predict X_test_scaled
                    knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbor[n], weights=weights)
                    y_pred = knn.fit(X_train, y_train).predict(X_test)
                    # add y_test and y_pred values to general real_y and predicted_y
                    for i in range(len(y_test)):
                        real_y.append(y_test[i])
                        predicted_y.append(y_pred[i])
                    # if high verbosity, calculate r and rmse at each split. Then print extra info
                    if verbosity_level>=2:
                        r_pearson,_=pearsonr(y_test,y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        average_r_pearson=average_r_pearson+r_pearson
                        average_rmse=average_rmse+rmse
                        f_out.write('Landscape %i . Adventurousness: %i . k-fold: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],counter_split,r_pearson,rmse))
                        f_out.write("%i test points: %s \n"  % (len(test_index), str(test_index)))
                        f_out.write("%i train points: %s \n" % (len(train_index),str(train_index)))
                        f_out.flush()
                    counter_split=counter_split+1
                # verbosity for average of splits
                if verbosity_level>=2:
                    average_r_pearson=average_r_pearson/counter_split
                    average_rmse=average_rmse/counter_split
                    f_out.write('Splits average r_pearson score: %f \n' % (average_r_pearson))
                    f_out.write('Splits average rmse score: %f \n' % (average_rmse))
                # calculate final r and rmse
                total_r_pearson,_ = pearsonr(real_y,predicted_y)
                total_mse = mean_squared_error(real_y, predicted_y)
                total_rmse = np.sqrt(total_mse)
            # For data sorted from old to new
            elif CV=='sort':
                # Use (1-'test_last_percentage') as training, and 'test_last_percentage' as test data
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_last_percentage,random_state=iseed,shuffle=False)
                # scale data
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                # fit kNN with (X_train, y_train), and predict X_test
                knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbor[n], weights=weights)
                y_pred = knn.fit(X_train, y_train).predict(X_test)
                # calculate final r and rmse
                total_r_pearson,_=pearsonr(y_test,y_pred)
                mse = mean_squared_error(y_test, y_pred)
                total_rmse = np.sqrt(mse)
                # print verbose info
                if verbosity_level>=2:
                    f_out.write("Train with first %i points \n" % (len(X_train)))
                    f_out.write("%s \n" % (str(X_train)))
                    f_out.write("Test with last %i points \n" % (len(X_test)))
                    f_out.write("%s \n" % (str(X_test)))
                    f_out.write('Landscape %i . Adventurousness: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],total_r_pearson,total_rmse))
            # Print last verbose info for kNN
            if verbosity_level>=1:
                f_out.write('Final r_pearson, rmse: %f, %f \n' % (total_r_pearson,total_rmse))
                f_out.flush()
            if n==0 or total_rmse<prev_total_rmse:
                if error_metric=='rmse': result=total_rmse
                final_k = n_neighbor[n]
                prev_total_rmse = total_rmse
        f_out.write("---------- \n")
        f_out.write("Final Optimum k: %i, rmse: %f \n" % (final_k,result))
    elif mode==2:
        for n in range(len(n_neighbor)):
            # initialize values
            real_y=[]
            predicted_y=[]
            provi_result = []
            # verbose info
            if verbosity_level>=1:
                f_out.write('## Start: "kNN" function \n')
                f_out.write('-------- \n')
                f_out.write('Perform k-NN \n')
                f_out.write('k= %i \n' % (n_neighbor[n]))
                f_out.write('weights %s \n' % (weights))
                f_out.write('iseed %s \n' % (iseed))
                f_out.write('-------- \n')
                f_out.flush()
            # assign train and test data
            X_train, X_test = Xtr, X
            y_train, y_test = ytr, y
            # scale data
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)        
            # fit kNN with (X_train_scaled, y_train) and predict X_test_scaled
            knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbor[n], weights=weights)
            # Train only at some steps
            time_taken1 = time()-start
            if t%t2_train_time==0:
                if verbosity_level>=1: f_out.write("At time %i, I am training new model\n" %(t))
                knn.fit(X_train_scaled, y_train)
                dump(knn, open('knn_%i.pkl' %(l), 'wb'))
                time_taken2 = time()-start
            else:
                if verbosity_level>=1: f_out.write("At time %i, I am reading previous trained model\n" %(t))
                knn=load(open('knn_%i.pkl' %(l), 'rb'))
                time_taken2 = time()-start
            if verbosity_level>=1: f_out.write("ML train took %0.4f seconds \n" %(time_taken2-time_taken1))
            ###############################################
            time_taken1 = time()-start
            y_pred=knn.predict(X_test_scaled)
            time_taken2 = time()-start
            f_out.write("ML predict took %0.4f seconds \n" %(time_taken2-time_taken1))
            # verbosity info
            if verbosity_level>=2: 
                f_out.write("X_train: %s\n" %(str(X_train)))
                f_out.write("y_train: %s\n" %(str(y_train)))
                f_out.write("X_test: %s\n" %(str(X_test)))
                f_out.write("y_test: %s\n" %(str(y_test)))
            # add y_test and y_pred values to general real_y and predicted_y
            for i in range(len(y_test)):
                real_y.append(y_test[i])
                predicted_y.append(y_pred[i])
            # calculate index of minimum predicted value (keep searching until a non-visited configuration is found)
            time_taken1 = time()-start
            #f_out.write("Croqueta X_train: %s\n" %(str(X_train)))
            #f_out.write("Croqueta X_test: %s\n" %(str(X_test)))
            for i in range(len(predicted_y)):
                min_index = predicted_y.index(sorted(predicted_y)[i])
                for k in range(len(X_train)):
                    counter_equal=0
                    for j in range(param):
                        #f_out.write("croqueta: k,j %i, %i\n" % (k,j))
                        if X_test[min_index][j] == X_train[k][j]: 
                            counter_equal=counter_equal+1
                            #f_out.write("croqueta %i, %i - I am increasing counter_equal by 1\n" %(k,j))
                    if counter_equal==param:
                        f_out.write("Croqueta: I am %s, %s, and am equal to a previous explored point \n" %(str(X_test[min_index]),str(predicted_y[min_index])))
                        break
                if counter_equal!=param:
                    f_out.write("Croqueta: I have encountered a minimum geometry that is new: %s, %s\n" %(str(X_test[min_index]),str(predicted_y[min_index])))
                    break
            time_taken2 = time()-start
            f_out.write("ML calculate next minimum  took %0.4f seconds \n" %(time_taken2-time_taken1))
            # print verbosity
            if verbosity_level>=2: 
                f_out.write("At index %i, predicted minimum value: %f\n" %(min_index, min(predicted_y)))
                f_out.write("At index %i, 'real' minimum value: %f\n" %(min_index, min(real_y)))
             # add predicted value to result
            for j in range(param):
                provi_result.append(X_test[min_index][j])
            provi_result.append(predicted_y[min_index])

            # calculate final r and rmse
            total_r_pearson,_ = pearsonr(real_y,predicted_y)
            total_mse = mean_squared_error(real_y, predicted_y)
            total_rmse = np.sqrt(total_mse)

            f_out.write("n_neighbor[n]: %i\n" %(n_neighbor[n]))
            f_out.write("total_rmse: %f\n" %(total_rmse))
            if n==0 or total_rmse<prev_total_rmse:
                #if error_metric=='rmse': result=total_rmse
                result = provi_result
                final_k = n_neighbor[n]
                prev_total_rmse = total_rmse
        f_out.write("Final k: %i, result: %s \n" % (final_k,str(result)))

    return result

# CALCULATE GBR #
def GBR(X,y,iseed,l,w,f_out):
    prev_total_rmse = 0
    gbr_counter = 0
    for gbr1 in range(len(GBR_n_estimators)):
        for gbr2 in range(len(GBR_learning_rate)):
            for gbr3 in range(len(GBR_max_depth)):
                for gbr4 in range(len(GBR_min_samples_split)):
                    for gbr5 in range(len(GBR_min_samples_leaf)):
                        iseed=iseed+1
                        if verbosity_level>=1: 
                            f_out.write('## Start: "GBR" function \n')
                            f_out.write('-------- \n')
                            f_out.write('Perform GBR\n')
                            f_out.write('cross_validation %i - fold\n' % (k_fold))
                            f_out.write('GBR criterion: %s\n' % (GBR_criterion))
                            f_out.write('Number of estimators: %i\n' % (GBR_n_estimators[gbr1]))
                            f_out.write('Learning rate: %f\n' % (GBR_learning_rate[gbr2]))
                            f_out.write('Tree max depth: %i\n' % (GBR_max_depth[gbr3]))
                            f_out.write('Min samples to split: %i\n' % (GBR_min_samples_split[gbr4]))
                            f_out.write('Min samples per leaf: %i\n' % (GBR_min_samples_leaf[gbr5]))
                            f_out.write('--------\n')
                            f_out.flush()
                    
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
                                GBR = GradientBoostingRegressor(criterion=GBR_criterion,n_estimators=GBR_n_estimators[gbr1],learning_rate=GBR_learning_rate[gbr2],max_depth=GBR_max_depth[gbr3],min_samples_split=GBR_min_samples_split[gbr4],min_samples_leaf=GBR_min_samples_leaf[gbr5])
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
                                    if verbosity_level>=2: 
                                        f_out.write('Landscape %i . Adventurousness: %i . k-fold: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],counter,r_pearson,rmse))
                                        f_out.write("%i test points: %s \n" % (len(test_index),str(test_index)))
                                    counter=counter+1
                                    average_r_pearson=average_r_pearson+r_pearson
                                    average_rmse=average_rmse+rmse
                            if CV=='kf':
                                average_r_pearson=average_r_pearson/k_fold
                                average_rmse=average_rmse/k_fold
                                if verbosity_level>=2: 
                                    f_out.write('k-fold average r_pearson score: %f \n' % (average_r_pearson))
                                    f_out.write('k-fold average rmse score: %f \n' % (average_rmse))
                            total_r_pearson,_ = pearsonr(real_y,predicted_y)
                            total_mse = mean_squared_error(real_y, predicted_y)
                            total_rmse = np.sqrt(total_mse)
                        elif CV=='sort':
                            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_last_percentage,random_state=iseed,shuffle=False)
                            GBR = GradientBoostingRegressor(criterion=GBR_criterion,n_estimators=GBR_n_estimators[gbr1],learning_rate=GBR_learning_rate[gbr2],max_depth=GBR_max_depth[gbr3],min_samples_split=GBR_min_samples_split[gbr4],min_samples_leaf=GBR_min_samples_leaf[gbr5])
                            y_pred = GBR.fit(X_train, y_train).predict(X_test)
                            total_r_pearson,_=pearsonr(y_test,y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            total_rmse = np.sqrt(mse)
                            if  verbosity_level>=1: 
                                f_out.write("Train with first %i points \n" % (len(X_train)))
                                f_out.write("%s \n" % (str(X_train)))
                                f_out.write("Test with last %i points \n" % (len(X_test)))
                                f_out.write("%s \n" % (str(X_test)))
                                f_out.write('Landscape %i . Adventurousness: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],total_r_pearson,total_rmse))
                        if verbosity_level>=1: 
                            f_out.write('Final r_pearson, rmse: %f, %f \n' % (total_r_pearson,total_rmse))
                            f_out.flush()
                        if gbr_counter==0 or total_rmse<prev_total_rmse:
                            final_gbr = []
                            if error_metric=='rmse': result=total_rmse
                            final_gbr = [GBR_n_estimators[gbr1],GBR_learning_rate[gbr2],GBR_max_depth[gbr3],GBR_min_samples_split[gbr4],GBR_min_samples_leaf[gbr5]]
                            f_out.write("New final_gbr: %s \n" % (str(final_gbr)))
                            prev_total_rmse = total_rmse
                        gbr_counter=gbr_counter+1
    f_out.write("---------- \n")
    f_out.write("Final Optimum GBR: %s, rmse: %f \n" % (str(final_gbr),result))
    return result


# CALCULATE GPR #
def GPR1(X,y,iseed,l,w,f_out,Xtr,ytr,mode,t):
    # initialize values
    iseed=iseed+1
    average_r=0.0
    average_r_pearson=0.0
    average_rmse=0.0
    real_y=[]
    predicted_y=[]
    counter_split=0
    if optimize_GPR_hyperparams==True:
        optimizer_GPR='fmin_l_bfgs_b'
    else:
        optimizer_GPR=None
    # CASE1: Calculate error metric
    if mode==1:
        # verbose info
        if verbosity_level>=1: 
            f_out.write('## Start: "GPR" function \n')
            f_out.write('-------- \n')
            f_out.write('Perform GPR\n')
            f_out.write('Cross_validation: %s\n' % (CV))
            f_out.write('Initial GPR_alpha %f: \n' % (GPR_alpha))
            f_out.write('-------- \n')
            f_out.flush()
        # assign splits to kf or and loo
        if CV=='kf':
            kf = KFold(n_splits=k_fold,shuffle=True,random_state=iseed)
            validation=kf.split(X)
        if CV=='loo':
            loo = LeaveOneOut()
            validation=loo.split(X)
        # For kf and loo
        if CV=='kf' or CV=='loo':
            # calculate r and rmse for each split
            for train_index, test_index in validation:
                # assign train and test data
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # scale data
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                ##### croqueta scaling ####
                f_out.write('X_train: %s\n' % (str(X_train)))
                f_out.write('X_train_scaled: %s\n' % (str(X_train_scaled)))
                f_out.write('y_train: %s\n' % (str(y_train)))
                f_out.write('X_test: %s\n' % (str(X_test)))
                f_out.write('X_test_scaled: %s\n' % (str(X_test_scaled)))
                f_out.write('y_test: %s\n' % (str(y_test)))
                ###################
                # fit GPR with (X_train_scaled, y_train) and predict X_test_scaled
                #kernel = GPR_A_RBF * RBF(length_scale=GPR_length_scale, length_scale_bounds=(1e-3, 1e+3)) + A_noise * WhiteKernel(noise_level=GPR_noise_level, noise_level_bounds=(1e-5, 1e+1))
                #GPR = GaussianProcessRegressor(kernel=kernel,alpha=GPR_alpha,normalize_y=True)
                #kernel = GPR_A_RBF * RBF(length_scale=GPR_length_scale, length_scale_bounds=(1e-3, 1e+3)) + WhiteKernel(noise_level=GPR_noise_level, noise_level_bounds=(1e-5, 1e+1))
                kernel = 1.0 * RBF(length_scale=GPR_length_scale, length_scale_bounds=(1e-3, 1e+3)) + 1.0 * WhiteKernel(noise_level=GPR_alpha, noise_level_bounds=(1e-20, 1e+1))
                GPR = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer_GPR, n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)
                y_pred = GPR.fit(X_train_scaled, y_train).predict(X_test_scaled)
                # add y_test and y_pred values to general real_y and predicted_y
                for i in range(len(y_test)):
                    real_y.append(y_test[i])
                    predicted_y.append(y_pred[i])
                # if high verbosity, calculate r and rmse at each split. Then print extra info
                if verbosity_level>=2: 
                    r_pearson,_=pearsonr(y_test,y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    average_r_pearson=average_r_pearson+r_pearson
                    average_rmse=average_rmse+rmse
                    f_out.write('Landscape %i . Adventurousness: %i . k-fold: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],counter_split,r_pearson,rmse))
                    f_out.write("%i test points: %s \n"  % (len(test_index), str(test_index)))
                    f_out.write("%i train points: %s \n" % (len(train_index),str(train_index)))
                    f_out.write('X_train: \n')
                    f_out.write('%s \n' % (str(X_train)))
                    f_out.write('y_train: \n')
                    f_out.write('%s \n' % (str(y_train)))
                    f_out.write('X_test: \n')
                    f_out.write('%s \n' % (str(X_test)))
                    f_out.write('y_test: \n')
                    f_out.write('%s \n' % (str(y_test)))
                    f_out.write('Converged kernel hyperparameters: %s \n' % (str(GPR.kernel_)))
                    f_out.write('Converged alpha: %s \n' % (str(GPR.alpha_)))
                    f_out.write('Parameters GPR: \n')
                    f_out.write('%s \n' % (str(GPR.get_params(deep=True))))
                    f_out.write('Parameters GPR kernel: \n')
                    f_out.write('%s \n' % (str(GPR.kernel_.get_params(deep=True))))
                    f_out.write('GPR X_train: \n')
                    f_out.write('%s \n' % (str(GPR.X_train_)))
                    f_out.write('GPR y_train: \n')
                    f_out.write('%s \n' % (str(GPR.y_train_)))
                    f_out.write('TEST X_train_scaled: \n')
                    f_out.write('%s \n' % (str(X_train_scaled)))
                    f_out.write('TEST X_test_scaled: \n')
                    f_out.write('%s \n' % (str(X_test_scaled)))
                    f_out.write('TEST y_test: \n')
                    f_out.write('%s \n' % (str(y_test)))
                    f_out.write('TEST y_pred: \n')
                    f_out.write('%s \n' % (str(y_pred)))
                    f_out.flush()
                counter_split=counter_split+1
            # verbosity for average of splits
            if verbosity_level>=2: 
                average_r_pearson=average_r_pearson/counter_split
                average_rmse=average_rmse/counter_split
                f_out.write('Splits average r_pearson score: %f \n' % (average_r_pearson))
                f_out.write('Splits average rmse score: %f \n' % (average_rmse))
            # calculate final r and rmse
            total_r_pearson,_ = pearsonr(real_y,predicted_y)
            total_mse = mean_squared_error(real_y, predicted_y)
            total_rmse = np.sqrt(total_mse)
        # For data sorted from old to new
        elif CV=='sort':
            # Use (1-'test_last_percentage') as training, and 'test_last_percentage' as test data
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_last_percentage,random_state=iseed,shuffle=False)
            # fit KRR with (X_train, y_train), and predict X_test
            #kernel = GPR_A_RBF * RBF(length_scale=GPR_length_scale, length_scale_bounds=(1e-3, 1e+3)) + A_noise * WhiteKernel(noise_level=GPR_noise_level, noise_level_bounds=(1e-5, 1e+1))
            #GPR = GaussianProcessRegressor(kernel=kernel,alpha=GPR_alpha,normalize_y=True)
            kernel = 1.0 * RBF(length_scale=GPR_length_scale, length_scale_bounds=(1e-3, 1e+3)) + 1.0 * WhiteKernel(noise_level=GPR_alpha, noise_level_bounds=(1e-20, 1e+1))
            GPR = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer_GPR, n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)
            # scale data
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            y_pred = GPR.fit(X_train_scaled, y_train).predict(X_test_scaled)
            # calculate final r and rmse
            total_r_pearson,_=pearsonr(y_test,y_pred)
            mse = mean_squared_error(y_test, y_pred)
            total_rmse = np.sqrt(mse)
            # print verbose info
            if verbosity_level>=2: 
                f_out.write("Train with first %i points \n" % (len(X_train)))
                f_out.write("%s \n" % (str(X_train)))
                f_out.write("Test with last %i points \n" % (len(X_test)))
                f_out.write("%s \n" % (str(X_test)))
                f_out.write('Landscape %i . Adventurousness: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],total_r_pearson,total_rmse))
                f_out.write('TEST X_train: \n')
                f_out.write('%s \n' % (str(X_train)))
                f_out.write('TEST y_train: \n')
                f_out.write('%s \n' % (str(y_train)))
                f_out.write('TEST X_test: \n')
                f_out.write('%s \n' % (str(X_test)))
                f_out.write('TEST y_test: \n')
                f_out.write('%s \n' % (str(y_test)))
                f_out.write('Converged kernel hyperparameters: %s \n' % (str(GPR.kernel_)))
                f_out.write('Converged alpha: %s \n' % (str(GPR.alpha_)))
                f_out.write('Parameters GPR: \n')
                f_out.write('%s \n' % (str(GPR.get_params(deep=True))))
                f_out.write('Parameters GPR kernel: \n')
                f_out.write('%s \n' % (str(GPR.kernel_.get_params(deep=True))))
                f_out.write('GPR X_train: \n')
                f_out.write('%s \n' % (str(GPR.X_train_)))
                f_out.write('GPR y_train: \n')
                f_out.write('%s \n' % (str(GPR.y_train_)))
                f_out.write('TEST X_train_scaled: \n')
                f_out.write('%s \n' % (str(X_train_scaled)))
                f_out.write('TEST X_test_scaled: \n')
                f_out.write('%s \n' % (str(X_test_scaled)))
                f_out.write('TEST y_test: \n')
                f_out.write('%s \n' % (str(y_test)))
                f_out.write('TEST y_pred: \n')
                f_out.write('%s \n' % (str(y_pred)))
                f_out.flush()
        # Print last verbose infor for GPR
        if verbosity_level>=1: 
            f_out.write('Final r_pearson, rmse: %f, %f \n' % (total_r_pearson,total_rmse))
            f_out.flush()
        if error_metric=='rmse': result=total_rmse
    # CASE 2: Predict minimum
    if mode==2:
        # initialize values
        real_y=[]
        predicted_y=[]
        result = []
        # verbose info
        if verbosity_level>=1: 
            f_out.write('## Start: "GPR" function \n')
            f_out.write('-------- \n')
            f_out.write('Perform GPR\n')
            f_out.write('Initial GPR_length_scale: %f \n' % (GPR_length_scale))
            f_out.write('Initial GPR_alpha: %f \n' % (GPR_alpha))
            #f_out.write('Initial GPR_n_restarts_optimizer: %f \n' % (GPR_n_restarts_optimizer))
            f_out.write('-------- \n')
            f_out.flush()
        # assign train and data tests
        X_train, X_test = Xtr, X
        y_train, y_test = ytr, y
        # scale data
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        ##### croqueta scaling ####
        f_out.write('X_train: %s\n' % (str(X_train)))
        f_out.write('X_train_scaled: %s\n' % (str(X_train_scaled)))
        f_out.write('y_train: %s\n' % (str(y_train)))
        f_out.write('X_test: %s\n' % (str(X_test)))
        f_out.write('X_test_scaled: %s\n' % (str(X_test_scaled)))
        f_out.write('y_test: %s\n' % (str(y_test)))
        ###################
        # fit GPR with (X_train_scaled, y_train) and predict X_test_scaled
        #kernel = GPR_A_RBF * RBF(length_scale=GPR_length_scale, length_scale_bounds=(1e-3, 1e+3)) + A_noise * WhiteKernel(noise_level=GPR_noise_level, noise_level_bounds=(1e-5, 1e+1))
        #GPR = GaussianProcessRegressor(kernel=kernel,alpha=GPR_alpha,normalize_y=True)
        kernel = 1.0 * RBF(length_scale=GPR_length_scale, length_scale_bounds=(1e-3, 1e+3)) + 1.0 * WhiteKernel(noise_level=GPR_alpha, noise_level_bounds=(1e-20, 1e+1))
        GPR = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer_GPR, n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)
        # Train only at some steps
        time_taken1 = time()-start
        if t%t2_train_time==0:
            if verbosity_level>=1: f_out.write("At time %i, I am training new model\n" %(t))
            GPR.fit(X_train_scaled, y_train)
            #dump(GPR, open('GPR.pkl', 'wb'))
            dump(GPR, open('GPR_%i.pkl' %(l), 'wb'))
            time_taken2 = time()-start
        else:
            if verbosity_level>=1: f_out.write("At time %i, I am reading previous trained model\n" %(t))
            #GPR=load(open('GPR.pkl', 'rb'))
            GPR=load(open('GPR_%i.pkl' %(l), 'rb'))
            time_taken2 = time()-start
        if verbosity_level>=1: f_out.write("ML train took %0.4f seconds \n" %(time_taken2-time_taken1))
        ##################################
        time_taken1 = time()-start
        y_pred = GPR.predict(X_test_scaled)
        time_taken2 = time()-start
        f_out.write("ML predict and fit took %0.4f seconds \n" %(time_taken2-time_taken1))
        # verbosity info
        if verbosity_level>=2: 
            f_out.write('X_train: \n')
            f_out.write('%s \n' % (str(X_train)))
            f_out.write('y_train: \n')
            f_out.write('%s \n' % (str(y_train)))
            f_out.write('X_test: \n')
            f_out.write('%s \n' % (str(X_test)))
            f_out.write('y_test: \n')
            f_out.write('%s \n' % (str(y_test)))
            f_out.write('Converged kernel hyperparameters: %s \n' % (str(GPR.kernel_)))
            f_out.write('Converged alpha: %s \n' % (str(GPR.alpha_)))
            f_out.write('Parameters GPR: \n')
            f_out.write('%s \n' % (str(GPR.get_params(deep=True))))
            f_out.write('Parameters GPR kernel: \n')
            f_out.write('%s \n' % (str(GPR.kernel_.get_params(deep=True))))
            f_out.write('X_train: \n')
            f_out.write('%s \n' % (str(GPR.X_train_)))
            f_out.write('y_train: \n')
            f_out.write('%s \n' % (str(GPR.y_train_)))
            f_out.write('X_train_scaled: \n')
            f_out.write('%s \n' % (str(X_train_scaled)))
            f_out.write('X_test_scaled: \n')
            f_out.write('%s \n' % (str(X_test_scaled)))
            f_out.write('y_test: \n')
            f_out.write('%s \n' % (str(y_test)))
            f_out.write('y_pred: \n')
            f_out.write('%s \n' % (str(y_pred)))
            f_out.flush()
        time_taken1 = time()-start
        # add y_test and y_pred values to general real_y and predicted_y
        for i in range(len(y_test)):
            real_y.append(y_test[i])
            predicted_y.append(y_pred[i])
        # calculate index of minimum predicted value
        min_index = predicted_y.index(min(predicted_y))
        # print verbosity
        if verbosity_level>=2: 
            f_out.write("At index %i, predicted minimum value: %f\n" %(min_index, min(predicted_y)))
            f_out.write("At index %i, real minimum value: %f\n" %(min_index, min(real_y)))
        # add predicted value to result
        for j in range(param):
            result.append(X_test[min_index][j])
        result.append(min(predicted_y))
        time_taken2 = time()-start
        f_out.write("ML calculate minimum took %0.4f seconds \n" %(time_taken2-time_taken1))
    return result



# CALCULATE GPR #
def GPR(hyperparams,X,y,iseed,l,w,f_out,Xtr,ytr,mode,t):
    # assign hyperparameters
    GPR_alpha,GPR_length_scale = hyperparams
    # initialize values
    iseed=iseed+1
    average_r=0.0
    average_r_pearson=0.0
    average_rmse=0.0
    real_y=[]
    predicted_y=[]
    counter_split=0
    # CASE1: Calculate error metric
    if mode==1:
        # verbose info
        if verbosity_level>=1: 
            f_out.write('## Start: "GPR" function \n')
            f_out.write('-------- \n')
            f_out.write('Perform GPR\n')
            f_out.write('Cross_validation %i - fold\n' % (k_fold))
            f_out.write('GPR alpha %f\n' % (GPR_alpha))
            f_out.write('GPR_length_scale %f\n' % (GPR_length_scale))
            f_out.write('-------- \n')
            f_out.flush()
        # assign splits for kf and loo
        if CV=='kf':
            kf = KFold(n_splits=k_fold,shuffle=True,random_state=iseed)
            validation=kf.split(X)
        if CV=='loo':
            loo = LeaveOneOut()
            validation=loo.split(X)
        # For kf and loo
        if CV=='kf' or CV=='loo':
            # calculate r and rmse for each split
            for train_index, test_index in validation:
                # assign train and test data
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # scale data
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                # fit GPR with (X_train, y_train), and predict X_test
                kernel = RBF(length_scale=GPR_length_scale, length_scale_bounds=(1e-3, 1e+3))
                GPR = GaussianProcessRegressor(kernel=kernel, alpha=GPR_alpha, optimizer=None, normalize_y=False,copy_X_train=True, random_state=None)
                y_pred = GPR.fit(X_train_scaled, y_train).predict(X_test_scaled)
                # add y_test and y_pred values to general real_y and predicted_y
                for i in range(len(y_test)):
                    real_y.append(y_test[i])
                    predicted_y.append(y_pred[i])
                # if high verbosity, calculate r and rmse at each split. Then print extra info
                if verbosity_level>=2: 
                    r_pearson,_=pearsonr(y_test,y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    average_r_pearson=average_r_pearson+r_pearson
                    average_rmse=average_rmse+rmse
                    f_out.write('Landscape %i . Adventurousness: %i . k-fold: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],counter_split,r_pearson,rmse))
                    f_out.write("%i test points: %s \n" % (len(test_index),str(test_index)))
                    f_out.write("%i train points: %s \n" % (len(train_index),str(train_index)))
                    f_out.write('X_train: \n')
                    f_out.write('%s \n' % (str(X_train)))
                    f_out.write('y_train: \n')
                    f_out.write('%s \n' % (str(y_train)))
                    f_out.write('X_test: \n')
                    f_out.write('%s \n' % (str(X_test)))
                    f_out.write('y_test: \n')
                    f_out.write('%s \n' % (str(y_test)))
                    f_out.write('Parameters GPR: \n')
                    f_out.write('%s \n' % (str(GPR.get_params(deep=True))))
                    f_out.write('X_train_scaled: \n')
                    f_out.write('%s \n' % (str(X_train_scaled)))
                    f_out.write('X_test_scaled: \n')
                    f_out.write('%s \n' % (str(X_test_scaled)))
                    f_out.write('y_test: \n')
                    f_out.write('%s \n' % (str(y_test)))
                    f_out.write('y_pred: \n')
                    f_out.write('%s \n' % (str(y_pred)))
                    f_out.flush()
                counter_split=counter_split+1
            # verbosity for average of splits
            if verbosity_level>=2: 
                average_r_pearson=average_r_pearson/counter_split
                average_rmse=average_rmse/counter_split
                f_out.write('Splits average r_pearson score: %f \n' % (average_r_pearson))
                f_out.write('Splits average rmse score: %f \n' % (average_rmse))
            # calculate final r and rmse
            total_r_pearson,_ = pearsonr(real_y,predicted_y)
            total_mse = mean_squared_error(real_y, predicted_y)
            total_rmse = np.sqrt(total_mse)
        # For data sorted from old to new
        elif CV=='sort':
            # Use (1-'test_last_percentage') as training, and 'test_last_percentage' as test data
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_last_percentage,random_state=iseed,shuffle=False)
            # scale data
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            # fit GPR with (X_train, y_train), and predict X_test
            kernel = RBF(length_scale=GPR_length_scale, length_scale_bounds=(1e-3, 1e+3))
            GPR = GaussianProcessRegressor(kernel=kernel, alpha=GPR_alpha, optimizer=None, normalize_y=False, copy_X_train=True, random_state=None)
            y_pred = GPR.fit(X_train_scaled, y_train).predict(X_test_scaled)
            # calculate final r and rmse
            total_r_pearson,_=pearsonr(y_test,y_pred)
            mse = mean_squared_error(y_test, y_pred)
            total_rmse = np.sqrt(mse)
            # print verbose info
            if  verbosity_level>=2: 
                f_out.write("Train with first %i points \n" % (len(X_train)))
                f_out.write("%s \n" % (str(X_train)))
                f_out.write("Test with last %i points \n" % (len(X_test)))
                f_out.write("%s \n" % (str(X_test)))
                f_out.write('Landscape %i . Adventurousness: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],total_r_pearson,total_rmse))
        # Print last verbose info for GPR
        if verbosity_level>=1: 
            f_out.write('Final r_pearson, rmse: %f, %f \n' % (total_r_pearson, total_rmse))
            f_out.flush()
        if error_metric=='rmse': result=total_rmse
    # CASE 2: Predict minimum
    if mode==2:
        # initialize values
        real_y=[]
        predicted_y=[]
        result = []
        # verbose info
        if verbosity_level>=1:
            f_out.write('## Start: "GPR" function \n')
            f_out.write('-------- \n')
            f_out.write('Perform GPR\n')
            f_out.write('Cross_validation %i - fold\n' % (k_fold))
            f_out.write('GPR alpha %f\n' % (GPR_alpha))
            f_out.write('GPR_length_scale %f\n' % (GPR_length_scale))
            f_out.write('-------- \n')
            f_out.flush()
        # assign train and data tests
        X_train, X_test = Xtr, X
        y_train, y_test = ytr, y
        # scale data
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # fit GPR with (X_train_scaled, y_train) and predict X_test_scaled
        kernel = RBF(length_scale=GPR_length_scale, length_scale_bounds=(1e-3, 1e+3))
        GPR = GaussianProcessRegressor(kernel=kernel, alpha=GPR_alpha, optimizer=None, normalize_y=False, copy_X_train=True, random_state=None)
        # Train only at some steps
        time_taken1 = time()-start
        if t%t2_train_time==0:
            if verbosity_level>=1: f_out.write("At time %i, I am training new model\n" %(t))
            GPR.fit(X_train_scaled, y_train)
            #dump(KRR, open('KRR.pkl', 'wb'))
            dump(GPR, open('GPR_%i.pkl' %(l), 'wb'))
            time_taken2 = time()-start
        else:
            if verbosity_level>=1: f_out.write("At time %i, I am reading previous trained model\n" %(t))
            #KRR=load(open('KRR.pkl', 'rb'))
            GPR=load(open('GPR_%i.pkl' %(l), 'rb'))
            time_taken2 = time()-start
        if verbosity_level>=1: f_out.write("ML train took %0.4f seconds \n" %(time_taken2-time_taken1))
        ###############################################
        y_pred = GPR.predict(X_test_scaled)
        # verbosity info
        if verbosity_level>=2:
            f_out.write('X_train: \n')
            f_out.write('%s \n' % (str(X_train)))
            f_out.write('y_train: \n')
            f_out.write('%s \n' % (str(y_train)))
            f_out.write('X_test: \n')
            f_out.write('%s \n' % (str(X_test)))
            f_out.write('y_test: \n')
            f_out.write('%s \n' % (str(y_test)))
            f_out.write('Parameters GPR: \n')
            f_out.write('%s \n' % (str(GPR.get_params(deep=True))))
            f_out.write('X_train_scaled: \n')
            f_out.write('%s \n' % (str(X_train_scaled)))
            f_out.write('X_test_scaled: \n')
            f_out.write('%s \n' % (str(X_test_scaled)))
            f_out.write('y_test: \n')
            f_out.write('%s \n' % (str(y_test)))
            f_out.write('y_pred: \n')
            f_out.write('%s \n' % (str(y_pred)))
            f_out.flush()
        # add y_test and y_pred values to general real_y and predicted_y
        for i in range(len(y_test)):
            real_y.append(y_test[i])
            predicted_y.append(y_pred[i])
        # calculate index of minimum predicted value
        min_index = predicted_y.index(min(predicted_y))
        # print verbosity
        if verbosity_level>=2:
            f_out.write("At index %i, predicted minimum value: %f\n" %(min_index, min(predicted_y)))
            f_out.write("At index %i, real minimum value: %f\n" %(min_index, min(real_y)))
        # add predicted value to result
        for j in range(param):
            result.append(X_test[min_index][j])
        result.append(min(predicted_y))
    return result


# CALCULATE KRR #
def KRR(hyperparams,X,y,iseed,l,w,f_out,Xtr,ytr,mode,t):
    # assign hyperparameters
    KRR_alpha,KRR_gamma = hyperparams
    # initialize values
    iseed=iseed+1
    average_r=0.0
    average_r_pearson=0.0
    average_rmse=0.0
    real_y=[]
    predicted_y=[]
    counter_split=0
    # CASE1: Calculate error metric
    if mode==1:
        # verbose info
        if verbosity_level>=1: 
            f_out.write('## Start: "KRR" function \n')
            f_out.write('-------- \n')
            f_out.write('Perform KRR\n')
            f_out.write('Cross_validation %i - fold\n' % (k_fold))
            f_out.write('KRR alpha %f\n' % (KRR_alpha))
            f_out.write('KRR gamma %f\n' % (KRR_gamma))
            f_out.write('KRR kernel %s\n' % (KRR_kernel))
            f_out.write('-------- \n')
            f_out.flush()
        # assign splits for kf and loo
        if CV=='kf':
            kf = KFold(n_splits=k_fold,shuffle=True,random_state=iseed)
            validation=kf.split(X)
        if CV=='loo':
            loo = LeaveOneOut()
            validation=loo.split(X)
        # For kf and loo
        if CV=='kf' or CV=='loo':
            # calculate r and rmse for each split
            for train_index, test_index in validation:
                # assign train and test data
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # scale data
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                # fit KRR with (X_train, y_train), and predict X_test
                KRR = KernelRidge(alpha=KRR_alpha,kernel=KRR_kernel,gamma=KRR_gamma)
                y_pred = KRR.fit(X_train_scaled, y_train).predict(X_test_scaled)
                # add y_test and y_pred values to general real_y and predicted_y
                for i in range(len(y_test)):
                    real_y.append(y_test[i])
                    predicted_y.append(y_pred[i])
                # if high verbosity, calculate r and rmse at each split. Then print extra info
                if verbosity_level>=2: 
                    r_pearson,_=pearsonr(y_test,y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    average_r_pearson=average_r_pearson+r_pearson
                    average_rmse=average_rmse+rmse
                    f_out.write('Landscape %i . Adventurousness: %i . k-fold: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],counter_split,r_pearson,rmse))
                    f_out.write("%i test points: %s \n" % (len(test_index),str(test_index)))
                    f_out.write("%i train points: %s \n" % (len(train_index),str(train_index)))
                    f_out.write('X_train: \n')
                    f_out.write('%s \n' % (str(X_train)))
                    f_out.write('y_train: \n')
                    f_out.write('%s \n' % (str(y_train)))
                    f_out.write('X_test: \n')
                    f_out.write('%s \n' % (str(X_test)))
                    f_out.write('y_test: \n')
                    f_out.write('%s \n' % (str(y_test)))
                    f_out.write('Parameters KRR: \n')
                    f_out.write('%s \n' % (str(KRR.get_params(deep=True))))
                    f_out.write('X_train_scaled: \n')
                    f_out.write('%s \n' % (str(X_train_scaled)))
                    f_out.write('X_test_scaled: \n')
                    f_out.write('%s \n' % (str(X_test_scaled)))
                    f_out.write('y_test: \n')
                    f_out.write('%s \n' % (str(y_test)))
                    f_out.write('y_pred: \n')
                    f_out.write('%s \n' % (str(y_pred)))
                    f_out.flush()
                counter_split=counter_split+1
            # verbosity for average of splits
            if verbosity_level>=2: 
                average_r_pearson=average_r_pearson/counter_split
                average_rmse=average_rmse/counter_split
                f_out.write('Splits average r_pearson score: %f \n' % (average_r_pearson))
                f_out.write('Splits average rmse score: %f \n' % (average_rmse))
            # calculate final r and rmse
            total_r_pearson,_ = pearsonr(real_y,predicted_y)
            total_mse = mean_squared_error(real_y, predicted_y)
            total_rmse = np.sqrt(total_mse)
        # For data sorted from old to new
        elif CV=='sort':
            # Use (1-'test_last_percentage') as training, and 'test_last_percentage' as test data
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_last_percentage,random_state=iseed,shuffle=False)
            # scale data
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            # fit KRR with (X_train, y_train), and predict X_test
            KRR = KernelRidge(alpha=KRR_alpha,kernel=KRR_kernel,gamma=KRR_gamma)
            y_pred = KRR.fit(X_train_scaled, y_train).predict(X_test_scaled)
            # calculate final r and rmse
            total_r_pearson,_=pearsonr(y_test,y_pred)
            mse = mean_squared_error(y_test, y_pred)
            total_rmse = np.sqrt(mse)
            # print verbose info
            if  verbosity_level>=2: 
                f_out.write("Train with first %i points \n" % (len(X_train)))
                f_out.write("%s \n" % (str(X_train)))
                f_out.write("Test with last %i points \n" % (len(X_test)))
                f_out.write("%s \n" % (str(X_test)))
                f_out.write('Landscape %i . Adventurousness: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],total_r_pearson,total_rmse))
        # Print last verbose info for KRR
        if verbosity_level>=1: 
            f_out.write('Final r_pearson, rmse: %f, %f \n' % (total_r_pearson, total_rmse))
            f_out.flush()
        if error_metric=='rmse': result=total_rmse
    # CASE 2: Predict minimum
    if mode==2:
        # initialize values
        real_y=[]
        predicted_y=[]
        result = []
        # verbose info
        if verbosity_level>=1:
            f_out.write('## Start: "KRR" function \n')
            f_out.write('-------- \n')
            f_out.write('Perform KRR\n')
            f_out.write('Cross_validation %i - fold\n' % (k_fold))
            f_out.write('KRR alpha %f\n' % (KRR_alpha))
            f_out.write('KRR gamma %f\n' % (KRR_gamma))
            f_out.write('KRR kernel %s\n' % (KRR_kernel))
            f_out.write('-------- \n')
            f_out.flush()
        # assign train and data tests
        X_train, X_test = Xtr, X
        y_train, y_test = ytr, y
        # scale data
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # fit KRR with (X_train_scaled, y_train) and predict X_test_scaled
        KRR = KernelRidge(alpha=KRR_alpha,kernel=KRR_kernel,gamma=KRR_gamma)
        # Train only at some steps
        time_taken1 = time()-start
        if t%t2_train_time==0:
            if verbosity_level>=1: f_out.write("At time %i, I am training new model\n" %(t))
            KRR.fit(X_train_scaled, y_train)
            #dump(KRR, open('KRR.pkl', 'wb'))
            dump(KRR, open('KRR_%i.pkl' %(l), 'wb'))
            time_taken2 = time()-start
        else:
            if verbosity_level>=1: f_out.write("At time %i, I am reading previous trained model\n" %(t))
            #KRR=load(open('KRR.pkl', 'rb'))
            KRR=load(open('KRR_%i.pkl' %(l), 'rb'))
            time_taken2 = time()-start
        if verbosity_level>=1: f_out.write("ML train took %0.4f seconds \n" %(time_taken2-time_taken1))
        ###############################################
        y_pred = KRR.predict(X_test_scaled)
        # verbosity info
        if verbosity_level>=2:
            f_out.write('X_train: \n')
            f_out.write('%s \n' % (str(X_train)))
            f_out.write('y_train: \n')
            f_out.write('%s \n' % (str(y_train)))
            f_out.write('X_test: \n')
            f_out.write('%s \n' % (str(X_test)))
            f_out.write('y_test: \n')
            f_out.write('%s \n' % (str(y_test)))
            f_out.write('Parameters KRR: \n')
            f_out.write('%s \n' % (str(KRR.get_params(deep=True))))
            f_out.write('X_train_scaled: \n')
            f_out.write('%s \n' % (str(X_train_scaled)))
            f_out.write('X_test_scaled: \n')
            f_out.write('%s \n' % (str(X_test_scaled)))
            f_out.write('y_test: \n')
            f_out.write('%s \n' % (str(y_test)))
            f_out.write('y_pred: \n')
            f_out.write('%s \n' % (str(y_pred)))
            f_out.flush()
        # add y_test and y_pred values to general real_y and predicted_y
        for i in range(len(y_test)):
            real_y.append(y_test[i])
            predicted_y.append(y_pred[i])
        # calculate index of minimum predicted value
        min_index = predicted_y.index(min(predicted_y))
        # print verbosity
        if verbosity_level>=2:
            f_out.write("At index %i, predicted minimum value: %f\n" %(min_index, min(predicted_y)))
            f_out.write("At index %i, real minimum value: %f\n" %(min_index, min(real_y)))
        # add predicted value to result
        for j in range(param):
            result.append(X_test[min_index][j])
        result.append(min(predicted_y))
    return result

# PLOT #
#def plot(flag,final_result_T):
def plot(flag,l,w,iseed,dim_list,G_list,X0,y0,X1,y1,results_per_walker_t1):
    #print('TEST plot, X0:')
    #print(X0)
    #print('TEST plot, X1:')
    #print(X1)
    if flag=='contour':
        print('Start: "plot(contour)"')
        # 2D contour plot
        pnt3d_1=plt.tricontour(dim_list[0],dim_list[1],G_list,20,linewidths=1,colors='k')
        plt.clabel(pnt3d_1,inline=1,fontsize=5)
        pnt3d_2=plt.tricontourf(dim_list[0],dim_list[1],G_list,100,cmap='Greys')
        cbar=plt.colorbar(pnt3d_2,pad=0.01)
        cbar.set_label("$G(\mathbf{x})$ (a.u.)",fontsize=12,labelpad=0)
        plt.xlabel('$x_1$ (a.u.)',fontsize=12)
        plt.ylabel('$x_2$ (a.u.)',fontsize=12)
        plt.axis([center_min,center_max,center_min,center_max])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('$S = %.2f$, $a = %s %s$. Seed: %i' %(float(S),adven[0],'\%',iseed),fontsize=15)
        nfile='_landscape'+str(l)
        file1='contour_2d' + nfile + '.png'
        plt.savefig(file1,format='png',dpi=600)
        print('save 2d map plot to %s' %file1)
        cbar.remove()
        plt.close()
    if flag=='t1_exploration':
        X1 = list(map(list, zip(*X1)))
        #tim=np.arange(t1_time)
        tim=np.arange(len(X1[0]))
        #tim_initial=np.arange(t0_time)
        pnt3d_1=plt.tricontour(dim_list[0],dim_list[1],G_list,20,linewidths=1,colors='k')
        plt.clabel(pnt3d_1,inline=1,fontsize=5)
        pnt3d_2=plt.tricontourf(dim_list[0],dim_list[1],G_list,100,cmap='Greys')
        pnt3d_3=plt.scatter(X1[0][:],X1[1][:],c=tim,cmap='inferno',s=50,linewidth=1,zorder=4,alpha=0.8)
        pnt3d_4=plt.scatter(X0[0][:],X0[1][:],c='black',s=50,linewidth=1,zorder=4,alpha=0.8)
        cbar_2=plt.colorbar(pnt3d_3,pad=0.06)
        #cbar_2=plt.colorbar(pnt3d_3)
        cbar_2.set_label("Time step \n", fontsize=12)
        cbar=plt.colorbar(pnt3d_2,pad=0.01)
        #cbar=plt.colorbar(pnt3d_2)
        cbar.set_label("$G(\mathbf{x})$ (a.u.)",fontsize=12,labelpad=0)
        plt.xlabel('$x_1$ (a.u.)',fontsize=12)
        plt.ylabel('$x_2$ (a.u.)',fontsize=12)
        plt.axis([center_min,center_max,center_min,center_max])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('$S = %.2f$, $a = %s %s$' %(float(S),adven[w],'\%'),fontsize=15)
        nfile='_landscape'+str(l)+'_walker'+str(w)
        file1='t1_exploration' + nfile + '.png'
        plt.savefig(file1,format='png',dpi=600)
        print('save 2d plot to %s' %file1)
        if w==Nwalkers-1:
            cbar.remove()
            cbar_2.remove()
        plt.close()
    if flag=='rmse':
        pntbox=plt.boxplot(results_per_walker_t1,patch_artist=True,labels=adven,showfliers=False)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel('Adventurousness (%)',fontsize=15)
        plt.ylabel('RMSE (a.u.)',fontsize=15)
        file1='rmse.png'
        plt.savefig(file1,format='png',dpi=600)
        print('save rmse box plot to %s' %file1,flush=True)
        plt.close()

        pntbox=plt.boxplot(results_per_walker_t1,patch_artist=True,labels=adven,showfliers=True)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel('Adventurousness (%)',fontsize=15)
        plt.ylabel('RMSE (a.u.)',fontsize=15)
        file1='rmse_with_outliers.png'
        plt.savefig(file1,format='png',dpi=600)
        print('save rmse box plot to %s' %file1,flush=True)
        plt.close()

##### END OTHER FUNCTIONS ######
################################################################################
# Measure initial time
start = time()
# Get initial values from input file
(dask_parallel, NCPU, verbosity_level, log_name,Nspf, S, iseed, param, center_min, center_max, grid_min, grid_max, grid_Delta, Nwalkers, adven, t1_time, d_threshold, t0_time, initial_sampling, ML, error_metric, CV, k_fold, test_last_percentage, n_neighbor, weights, GBR_criterion, GBR_n_estimators, GBR_learning_rate, GBR_max_depth, GBR_min_samples_split, GBR_min_samples_leaf, GPR_alpha, GPR_length_scale, GPR_alpha_lim, GPR_length_scale_lim, KRR_alpha, KRR_kernel,  KRR_gamma, optimize_KRR_hyperparams, optimize_GPR_hyperparams, KRR_alpha_lim, KRR_gamma_lim, allowed_initial_sampling, allowed_CV, allowed_ML, allowed_ML, allowed_error_metric, width_min, width_max, Amplitude_min, Amplitude_max, N, t2_time, allowed_verbosity_level, t2_ML, allowed_t2_ML, t2_exploration, t1_analysis, diff_popsize, diff_tol, t2_train_time, calculate_grid, grid_name,plot_t1_exploration,plot_contour_map,plot_t1_error_metric,initial_spf) = read_initial_values(input_file_name)
# Run main program
main(iseed)
# Measure and print final time
time_taken = time()-start
print ('Process took %0.4f seconds' %time_taken)
