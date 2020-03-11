#!/usr/bin/env python3
# Marcos del Cueto
# Department of Chemistry and MIF, University of Liverpool
#################################################################################
import ast
import sys
import dask
import random
import math
import statistics
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from time import time
from dask import delayed
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
from pickle import dump,load
#################################################################################
######   START CUSTOMIZABLE PARAMETERS ########
input_file_name = 'input_MLL.txt'      # name of input file
######   END CUSTOMIZABLE PARAMETERS   ######
#################################################################################

#################################################################################
###### START MAIN ######
def main(iseed):
    # Check that input values are OK
    check_input_values()
    # Initialize results array
    results_t1_per_Nspf=[]
    results_t2_per_Nspf=[]
    # Calculate results for each landscape (may use dask to run each landscape in a CPU in parallel)
    if is_dask==True:
        for l in range(Nspf):
            iseed=iseed+l
            (provi_result_t1,provi_result_t2)=delayed(MLL,nout=2)(iseed,l)
            results_t1_per_Nspf.append(provi_result_t1)
            results_t2_per_Nspf.append(provi_result_t2)
        results_t1_per_Nspf=dask.compute(results_t1_per_Nspf,scheduler='processes',num_workers=NCPU)
        results_t2_per_Nspf=dask.compute(results_t2_per_Nspf,scheduler='processes',num_workers=NCPU)
        results_t1_per_Nspf=results_t1_per_Nspf[0]
        results_t2_per_Nspf=results_t2_per_Nspf[0]
    elif is_dask==False:
        for l in range(Nspf):
            iseed=iseed+l
            (provi_result_t1,provi_result_t2)=MLL(iseed,l)
            results_t1_per_Nspf.append(provi_result_t1)
            results_t2_per_Nspf.append(provi_result_t2)
    # Transpose results_per_Nspf, to get results per walker
    if t1_analysis    == True: results_per_walker_t1=[list(i) for i in zip(*results_t1_per_Nspf)]
    if t2_exploration == True: results_per_walker_t2=[list(i) for i in zip(*results_t2_per_Nspf)]
    # Print final results
    print('--- Final results ---',flush=True)
    for i in range(Nwalkers):
        print('-- Adventurousness: %6.1f --' %(adven[i]),flush=True)
        if t1_analysis == True:
            print('-- t1 analysis')
            print('- RMSE:',results_per_walker_t1[i][:],flush=True)
            print('- RMSE Mean: %f' %(statistics.mean(results_per_walker_t1[i])),flush=True)
            print('- RMSE Median: %f' %(statistics.median(results_per_walker_t1[i])),flush=True)
        if t2_exploration == True:
            print('-- t2 exploration')
            print('- [ML_gain_pred, ML_gain_real, error_rel_ML]: %s' %(str(results_per_walker_t2[i])),flush=True)
            ML_gain_pred = [item[0] for item in results_per_walker_t2[i]]
            ML_gain_real = [item[1] for item in results_per_walker_t2[i]]
            error_rel_ML = [item[2] for item in results_per_walker_t2[i]]
            print('- ML_gain_pred Mean: %f' %(statistics.mean(ML_gain_pred)),flush=True)
            print('- ML_gain_pred Median: %f' %(statistics.median(ML_gain_pred)),flush=True)
            print('- ML_gain_real Mean: %f' %(statistics.mean(ML_gain_real)),flush=True)
            print('- ML_gain_real Median: %f' %(statistics.median(ML_gain_real)),flush=True)
            print('- error_rel_ML Mean: %f' %(statistics.mean(error_rel_ML)),flush=True)
            print('- error_rel_ML Median: %f' %(statistics.median(error_rel_ML)),flush=True)
        print('')

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
    verbosity_level = ast.literal_eval(var_value[var_name.index('verbosity_level')])
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
    GPR_A_RBF = ast.literal_eval(var_value[var_name.index('GPR_A_RBF')])
    GPR_length_scale = ast.literal_eval(var_value[var_name.index('GPR_length_scale')])
    GPR_noise_level = ast.literal_eval(var_value[var_name.index('GPR_noise_level')])
    KRR_alpha = ast.literal_eval(var_value[var_name.index('KRR_alpha')])
    KRR_kernel = ast.literal_eval(var_value[var_name.index('KRR_kernel')])
    KRR_gamma = ast.literal_eval(var_value[var_name.index('KRR_gamma')])
    optimize_KRR_hyperparams = ast.literal_eval(var_value[var_name.index('optimize_KRR_hyperparams')])
    optimize_GPR_hyperparams = ast.literal_eval(var_value[var_name.index('optimize_GPR_hyperparams')])
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

    width_min=S                                     # Minimum width of each Gaussian function
    width_max=1.0/3.0                               # Maximum width of each Gaussian function
    Amplitude_min=0.0                               # Minimum amplitude of each Gaussian function
    Amplitude_max=1.0                               # Maximum amplitude of each Gaussian function
    N=int(round((1/(S**param))))                    # Number of Gaussian functions of a specific landscape
    if iseed==None: 
        iseed=random.randrange(2**30-1) # If no seed is specified, choose a random one

    return (is_dask,NCPU,verbosity_level,log_name,Nspf,S,iseed,param,center_min,center_max,grid_min,grid_max,grid_Delta,Nwalkers,adven,t1_time,d_threshold,t0_time,initial_sampling,ML,error_metric,CV,k_fold,test_last_percentage,n_neighbor,weights,GBR_criterion,GBR_n_estimators,GBR_learning_rate,GBR_max_depth,GBR_min_samples_split,GBR_min_samples_leaf,GPR_A_RBF,GPR_length_scale,GPR_noise_level,KRR_alpha,KRR_kernel,KRR_gamma,optimize_KRR_hyperparams,optimize_GPR_hyperparams,KRR_alpha_lim,KRR_gamma_lim,allowed_initial_sampling,allowed_CV,allowed_ML,allowed_ML,allowed_error_metric,width_min,width_max,Amplitude_min,Amplitude_max,N,t2_time,allowed_verbosity_level,t2_ML,allowed_t2_ML,t2_exploration,t1_analysis,diff_popsize,diff_tol)

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
    dim_list, G_list, Ngrid, max_G = generate_grid(iseed,l,f_out)
    time_taken2 = time()-start
    if verbosity_level>=1:
        f_out.write("Generate grid took %0.4f seconds\n" %(time_taken2-time_taken1))
    # For each walker
    for w in range(Nwalkers):
        # Step 1) Perform t1 exploration
        time_taken1 = time()-start
        X1,y1,unique_t1 = explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,t0_time,t1_time,0,None,None,False)
        time_taken2 = time()-start
        if verbosity_level>=1:
            f_out.write("t1 exploration took %0.4f seconds\n" %(time_taken2-time_taken1))
        if t1_analysis == True:
        # Step 2A) Calculate error_metric
            time_taken1 = time()-start
            if ML=='kNN': error_metric_result=kNN(X1,y1,iseed,l,w,f_out,None,None,1,None)
            if ML=='GBR': error_metric_result=GBR(X1,y1,iseed,l,w,f_out)
            if ML=='GPR': error_metric_result=GPR(X1,y1,iseed,l,w,f_out,None,None,1)
            if ML=='KRR':
                hyperparams=[KRR_gamma,KRR_alpha]
                if optimize_KRR_hyperparams == False:
                    error_metric_result=KRR(hyperparams,X1,y1,iseed,l,w,f_out,None,None,1)
                else:
                    mini_args=(X1,y1,iseed,l,w,f_out,None,None,1)
                    bounds = [KRR_alpha_lim]+[KRR_gamma_lim]
                    solver=differential_evolution(KRR,bounds,args=mini_args,popsize=diff_popsize,tol=diff_tol)
                    best_hyperparams = solver.x
                    best_rmse = solver.fun
                    if verbosity_level>=1: f_out.write("Best hyperparameters: %s \n" %str(best_hyperparams))
                    if verbosity_level>=1: f_out.write("Best rmse: %f \n"  %best_rmse)
                    if verbosity_level>=1: f_out.flush()
                    error_metric_result = best_rmse
            error_metric_list.append(error_metric_result)
            result1 = error_metric_list
            time_taken2 = time()-start
            if verbosity_level>=1:
                f_out.write("t1 analysis took %0.4f seconds\n" %(time_taken2-time_taken1))
        # Step 2B) Perform t2 exploration
        if t2_exploration == True:
            # Step 2.B.1) Perform t2 exploration with random biased explorer
            time_taken1 = time()-start
            X2a,y2a,unique_t2a = explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,0,unique_t1,t2_time,X1,y1,False)
            time_taken2 = time()-start
            if verbosity_level>=1:
                f_out.write("t2 standard exploration took %0.4f seconds\n" %(time_taken2-time_taken1))
            # Step 2.B.2) Perform t2 exploration with ML explorer
            time_taken1 = time()-start
            X2b,y2b,unique_t2b = explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,0,unique_t1,t2_time,X1,y1,True)
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
            MLgain_pred = ( min(y2a) - min(y2b))/abs(min(y2a))
            MLgain_real = ( min(y2a) - y_real)/abs(min(y2a))
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
                f_out.write("Predicted relative MLgain: %.3f%s \n" %(MLgain_pred*100, "%"))
                f_out.write("Real relative MLgain: %.3f%s \n" %(MLgain_real*100, "%"))
                f_out.write("ML prediction relative error: %.3f%s \n" %(error_ML*100, "%"))
                f_out.write("################ \n")
                f_out.flush()

            ML_benefits=[]
            ML_benefits.append(MLgain_pred)
            ML_benefits.append(MLgain_real)
            ML_benefits.append(error_ML)
            if verbosity_level>=1: 
                f_out.write("For each Nwalker: %s\n" %(str(ML_benefits)))
                f_out.flush()
            ML_benefits_list.append(ML_benefits)
            result2=ML_benefits_list
    time_taken2 = time()-start
    if verbosity_level>=1:
        f_out.write("Rest of MLL took %0.4f seconds\n" %(time_taken2-time_taken1))
    if verbosity_level>=1: 
        f_out.write("I am returning these values: %s, %s\n" %(str(result1), str(result2)))
        f_out.close()
    return (result1, result2)

def generate_grid(iseed,l,f_out):
    Amplitude      = []
    center_N       = [[] for i in range(N)]
    width_N        = [[] for i in range(N)]
    dim_list       = [[] for i in range(param)]
    G_list         = []
    if verbosity_level>=1: 
        f_out.write('## Start: "generate_grid" function \n')
        f_out.write("########################### \n")
        f_out.write("##### Landscape', %i '##### \n" % (l))
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
        f_out.write("%4s %14s %22s %34s \n" % ("N","Amplitude","Center","Width"))
        for i in range(len(Amplitude)):
            line1 = []
            line2 = []
            for j in range(param):
                line1.append((center_N[i][j]))
                line2.append((width_N[i][j]))
            f_out.write("%4i %2s %10.6f %2s %s %2s %s \n" % (i, "", Amplitude[i],"",str(line1),"",str(line2)))
        f_out.flush()
    # CALCULATE G GRID #
    counter=0
    if verbosity_level>=2: f_out.write("%8s %11s %15s \n" % ("i","x","G"))
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
        if verbosity_level>=2: 
            f_out.write("%8i %2s %s \n" % (counter,"",str(line)))
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

    return dim_list, G_list, Ngrid, max_G

def check_input_values():
    if type(is_dask) != bool:
        print ('INPUT ERROR: is_dask should be boolean, but is:', is_dask)
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
    print("\n",flush=True)
    print('##### START PRINT INPUT  #####',flush=True)
    print('##############################')
    print('# General Landscape parameters')
    print('##############################')
    print('### Parallel computing ###',flush=True)
    print('is_dask =',is_dask,flush=True)
    print('NCPU',NCPU,flush=True)
    print('### Verbose ###',flush=True)
    print('verbosity_level =',verbosity_level,flush=True)
    print('allowed_verbosity_level =',allowed_verbosity_level,flush=True)
    print('log_name',log_name,flush=True)
    print('### Landscape parameters ###',flush=True)
    print('Nspf =',Nspf,flush=True)
    print('S =',S,flush=True)
    print('iseed =',iseed,flush=True)
    print('param =',param,flush=True)
    print('center_min =',center_min,flush=True)
    print('center_max =',center_max,flush=True)
    print('### Grid parameters ###',flush=True)
    print('grid_min =',grid_min,flush=True)
    print('grid_max =',grid_max,flush=True)
    print('grid_Delta =',grid_Delta,flush=True)
    print('##############################')
    print('# T1 exploration parameters')
    print('##############################')
    print('Nwalkers =',Nwalkers,flush=True)
    print('adven =',adven,flush=True)
    print('t0_time =',t0_time,flush=True)
    print('t1_time =',t1_time,flush=True)
    print('d_threshold =',d_threshold,flush=True)
    print('initial_sampling =',initial_sampling,flush=True)
    print('allowed_initial_sampling =',allowed_initial_sampling,flush=True)
    print('##############################')
    print('# T2 exploration parameters')
    print('##############################')
    print('t2_exploration =',t2_exploration,flush=True)
    print('t2_time =',t2_time,flush=True)
    print('t2_ML =',t2_ML,flush=True)
    print('allowed_t2_ML =',allowed_t2_ML,flush=True)
    print('##############################')
    print('# Error metric parameters')
    print('##############################')
    print('t1_analysis =',t1_analysis,flush=True)
    print('error_metric =',error_metric,flush=True)
    print('allowed_error_metric =',allowed_error_metric,flush=True)
    print('ML =',ML,flush=True)
    print('allowed_ML =',allowed_ML,flush=True)
    print('CV =',CV,flush=True)
    print('allowed_CV =',allowed_CV,flush=True)
    print('k_fold =',k_fold,flush=True)
    print('test_last_percentage =',test_last_percentage,flush=True)
    if ML=='kNN':
        print('### kNN parameters ###')
        print('n_neighbor =',n_neighbor,flush=True)
        print('weights =',weights,flush=True)
    if ML=='GBR':
        print('### GBR parameters ###')
        print('GBR_criterion =',GBR_criterion,flush=True)
        print('GBR_n_estimators =',GBR_n_estimators,flush=True)
        print('GBR_learning_rate =',GBR_learning_rate,flush=True)
        print('GBR_max_depth =',GBR_max_depth,flush=True)
        print('GBR_min_samples_split =',GBR_min_samples_split,flush=True)
        print('GBR_min_samples_leaf =',GBR_min_samples_leaf,flush=True)
    if ML=='GPR':
        print('### GPR parameters ###')
        print('GPR_A_RBF =',GPR_A_RBF,flush=True)
        print('GPR_length_scale =',GPR_length_scale,flush=True)
        print('GPR_noise_level =',GPR_noise_level,flush=True)
    if ML=='KRR':
        print('### KRR parameters ###')
        print('KRR_alpha =',KRR_alpha,flush=True)
        print('KRR_kernel =',KRR_kernel,flush=True)
        print('KRR_gamma =',KRR_gamma,flush=True)
        print('optimize_KRR_hyperparams =',optimize_KRR_hyperparams,flush=True)
        print('KRR_alpha_lim =',KRR_alpha_lim,flush=True)
        print('KRR_gamma_lim =',KRR_gamma_lim,flush=True)
    print('##############################')
    print('# Differential evolution parameters')
    print('diff_popsize =', diff_popsize)
    print('diff_tol =', diff_tol)
    print('##############################')
    print('### Calculated parameters ###')
    print('width_min =',width_min,flush=True)
    print('width_max =',width_max,flush=True)
    print('Amplitude_min =',Amplitude_min,flush=True)
    print('Amplitude_max =',Amplitude_max,flush=True)
    print('N =',N,flush=True)
    print('#####   END PRINT INPUT  #####',flush=True)
    print("\n",flush=True)

def explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G,t0,t1,t2,Xi,yi,ML_explore):
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
    ### Perform t1 and t2 standard exploration ###
    if ML_explore == False:

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
                f_out.write("Consider nearby points: \n")
                f_out.write("%6s %11s %19s %12s %12s \n" % ("i","x","G","Prob","distance"))
                f_out.flush()
            # Check for inconsistencies
            for i in range(param):
                if minimum_path_x[i][draw] != dim_list[i][draw_in_grid]:
                    print("STOP - ERROR: minimum_path not equal to dum_list (maybe more than 1 point with that value in grid)",flush=True)
                    print("Selected point draw:", minimum_path_x[:][draw],minimum_path_G[draw],flush=True)
                    print("Selected point draw in grid:", dim_list[:][draw_in_grid],G_list[draw_in_grid],flush=True)
                    sys.exit()
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
                f_out.write("Number of points considered: %i \n" % (len(range((P*2+1)**param))))
                f_out.write("Points within threshold: %f \n" % int(round((prob_sum))))
                f_out.flush()
            # renormalize probabilities
            for i in range(counter3):
                prob[i]=prob[i]/prob_sum
            # choose neighbor to draw
            draw=int(choice(range((P*2+1)**param),size=1,p=prob))
            # add draw's X and G values to path_x and path_G
            for i in range(param):
                walker_x[i]=neighbor_walker[i][draw]
                path_x[i].append(walker_x[i])
            path_G.append(neighbor_G[draw])
            list_t.append(t)
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
                    prob_sum=0.0
                    # get coordinates of kth point in SPF grid
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
            if t2_ML=='GPR': min_point=GPR(x_bubble,y_bubble,iseed,l,w,f_out,path_x,path_G,2)
            if t2_ML=='KRR':
                hyperparams=[KRR_gamma,KRR_alpha]
                if optimize_KRR_hyperparams == False:
                    min_point=KRR(hyperparams,x_bubble,y_bubble,iseed,l,w,f_out,path_x,path_G,2)
                else:
                    mini_args=(path_x,path_G,iseed,l,w,f_out,None,None,1) # get rmse fitting previous points
                    bounds = [KRR_alpha_lim]+[KRR_gamma_lim]
                    solver=differential_evolution(KRR,bounds,args=mini_args,popsize=diff_popsize,tol=diff_tol)
                    best_hyperparams = solver.x
                    best_rmse = solver.fun
                    if verbosity_level>=1: 
                        f_out.write("Best hyperparameters: %s \n" %str(best_hyperparams))
                        f_out.write("Best rmse: %f \n" %best_rmse)
                        f_out.flush()
                    hyperparams=[best_hyperparams[0],best_hyperparams[1]]
                    min_point=KRR(hyperparams,x_bubble,y_bubble,iseed,l,w,f_out,path_x,path_G,2)
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
    # CASE1: Calculate error metric
    if mode==1:
        # verbose info
        if verbosity_level>=1: 
            f_out.write('## Start: "kNN" function \n')
            f_out.write('-------- \n')
            f_out.write('Perform k-NN \n')
            f_out.write('k= %i \n' % (n_neighbor))
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
                knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbor, weights=weights)
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
            knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbor, weights=weights)
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
        if error_metric=='rmse': result=total_rmse
    elif mode==2:
        # initialize values
        real_y=[]
        predicted_y=[]
        result = []
        # verbose info
        if verbosity_level>=1:
            f_out.write('## Start: "kNN" function \n')
            f_out.write('-------- \n')
            f_out.write('Perform k-NN \n')
            f_out.write('k= %i \n' % (n_neighbor))
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
        knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbor, weights=weights)
        ############################################## Test train only at some steps
        #time_taken1 = time()-start
        #if t==0:
            #f_out.write("Croqueta, At time %i, I am training with: %s, %s \n" %(t,str(X_train),str(y_train)))
            #knn.fit(X_train_scaled, y_train)
            #dump(knn, open('knn.pkl', 'wb'))
            #time_taken2 = time()-start

        #if t>0:
            #knn=load(open('knn.pkl', 'rb'))
            #time_taken2 = time()-start
        #f_out.write("ML train took %0.4f seconds \n" %(time_taken2-time_taken1))
        ################################################
        knn.fit(X_train_scaled, y_train)
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
        f_out.write("Croqueta X_train: %s\n" %(str(X_train)))
        f_out.write("Croqueta X_test: %s\n" %(str(X_test)))
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
            result.append(X_test[min_index][j])
        result.append(predicted_y[min_index])
    return result

# CALCULATE GBR #
def GBR(X,y,iseed,l,w,f_out):
    iseed=iseed+1
    if verbosity_level>=1: 
        f_out.write('## Start: "GBR" function \n')
        f_out.write('-------- \n')
        f_out.write('Perform GBR\n')
        f_out.write('cross_validation %i - fold\n' % (k_fold))
        f_out.write('GBR criterion: %s\n' % (GBR_criterion))
        f_out.write('Number of estimators: %i\n' % (GBR_n_estimators))
        f_out.write('Learning rate: %f\n' % (GBR_learning_rate))
        f_out.write('Tree max depth: %i\n' % (GBR_max_depth))
        f_out.write('Min samples to split: %i\n' % (GBR_min_samples_split))
        f_out.write('Min samples per leaf: %i\n' % (GBR_min_samples_leaf))
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
        GBR = GradientBoostingRegressor(criterion=GBR_criterion,n_estimators=GBR_n_estimators,learning_rate=GBR_learning_rate,max_depth=GBR_max_depth,min_samples_split=GBR_min_samples_split,min_samples_leaf=GBR_min_samples_leaf)
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
        f_out.write('Final r_pearson score: %f \n' % (total_r_pearson))
        f_out.write('Final rmse score: %f \n' % (total_rmse))
        f_out.flush()
    if error_metric=='rmse': result=total_rmse
    return result


# CALCULATE GPR #
def GPR(X,y,iseed,l,w,f_out,Xtr,ytr,mode):
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
            f_out.write('Initial GPR_A_RBF %f: \n' % (GPR_A_RBF))
            f_out.write('Initial GPR_noise_level: %f: \n' % (GPR_noise_level))
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
                # fit GPR with (X_train_scaled, y_train) and predict X_test_scaled
                #kernel = GPR_A_RBF * RBF(length_scale=GPR_length_scale, length_scale_bounds=(1e-3, 1e+3)) + A_noise * WhiteKernel(noise_level=GPR_noise_level, noise_level_bounds=(1e-5, 1e+1))
                #GPR = GaussianProcessRegressor(kernel=kernel,alpha=GPR_alpha,normalize_y=True)
                kernel = GPR_A_RBF * RBF(length_scale=GPR_length_scale, length_scale_bounds=(1e-3, 1e+3)) + WhiteKernel(noise_level=GPR_noise_level, noise_level_bounds=(1e-5, 1e+1))
                GPR = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, optimizer=optimizer_GPR, n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)
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
            kernel = GPR_A_RBF * RBF(length_scale=GPR_length_scale, length_scale_bounds=(1e-3, 1e+3)) + WhiteKernel(noise_level=GPR_noise_level, noise_level_bounds=(1e-5, 1e+1))
            GPR = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, optimizer=optimizer_GPR, n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)
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
            f_out.write('Initial GPR_A_RBF %f \n' % (GPR_A_RBF))
            f_out.write('Initial GPR_length_scale: %f \n' % (GPR_length_scale))
            f_out.write('Initial GPR_noise_level: %f \n' % (GPR_noise_level))
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
        #kernel = GPR_A_RBF * RBF(length_scale=GPR_length_scale, length_scale_bounds=(1e-3, 1e+3)) + A_noise * WhiteKernel(noise_level=GPR_noise_level, noise_level_bounds=(1e-5, 1e+1))
        #GPR = GaussianProcessRegressor(kernel=kernel,alpha=GPR_alpha,normalize_y=True)
        kernel = GPR_A_RBF * RBF(length_scale=GPR_length_scale, length_scale_bounds=(1e-3, 1e+3)) + WhiteKernel(noise_level=GPR_noise_level, noise_level_bounds=(1e-5, 1e+1))
        GPR = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, optimizer=optimizer_GPR, n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)
        time_taken1 = time()-start
        y_pred = GPR.fit(X_train_scaled,y_train).predict(X_test_scaled)
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
def KRR(hyperparams,X,y,iseed,l,w,f_out,Xtr,ytr,mode):
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
                    f_out.write('Parameters GPR: \n')
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
        # fit GPR with (X_train_scaled, y_train) and predict X_test_scaled
        KRR = KernelRidge(alpha=KRR_alpha,kernel=KRR_kernel,gamma=KRR_gamma)
        y_pred = KRR.fit(X_train_scaled, y_train).predict(X_test_scaled)
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
# Measure initial time
start = time()
# Get initial values from input file
(is_dask,NCPU,verbosity_level,log_name,Nspf,S,iseed,param,center_min,center_max,grid_min,grid_max,grid_Delta,Nwalkers,adven,t1_time,d_threshold,t0_time,initial_sampling,ML,error_metric,CV,k_fold,test_last_percentage,n_neighbor,weights,GBR_criterion,GBR_n_estimators,GBR_learning_rate,GBR_max_depth,GBR_min_samples_split,GBR_min_samples_leaf,GPR_A_RBF,GPR_length_scale,GPR_noise_level,KRR_alpha,KRR_kernel,KRR_gamma,optimize_KRR_hyperparams,optimize_GPR_hyperparams,KRR_alpha_lim,KRR_gamma_lim,allowed_initial_sampling,allowed_CV,allowed_ML,allowed_ML,allowed_error_metric,width_min,width_max,Amplitude_min,Amplitude_max,N,t2_time,allowed_verbosity_level,t2_ML,allowed_t2_ML,t2_exploration,t1_analysis,diff_popsize,diff_tol) = read_initial_values(input_file_name)
# Run main program
main(iseed)
# Measure and print final time
time_taken = time()-start
print ('Process took %0.4f seconds' %time_taken)
