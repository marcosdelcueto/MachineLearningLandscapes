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
from numpy.random import choice
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
from sklearn import neighbors, preprocessing
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.model_selection import LeaveOneOut, train_test_split

#################################################################################
######   START CUSTOMIZABLE PARAMETERS ########
input_file_name = 'input_MLL.inp'      # name of input file
######   END CUSTOMIZABLE PARAMETERS   ######
#################################################################################

#################################################################################
###### START MAIN ######
def main():
    print('#############################')
    print('### CALLING MAIN FUNCTION ###')
    print('#############################')
    # Calculation just to generate SPF grid
    if calculate_grid == True:
        # Use paralellization: a SPF per CPU
        if dask_parallel==True:
            dummy_list=[]
            for l in range(Nspf):
                dummy = delayed(generate_grid)(l)
                dummy_list.append(dummy)
            result=dask.compute(dummy_list,scheduler='processes',num_workers=NCPU)
        # Calculate SPFs in serial (all in 1 CPU)
        else:
            for l in range(Nspf):
                Ngrid = generate_grid(l)
    # Explore SPF
    if t1_analysis == True:
        # Initialize results array
        results_t1_per_Nspf=[]
        results_t2_per_Nspf=[]
        # Calculate results for each landscape (may use dask to run each landscape in a CPU in parallel)
        if dask_parallel==True:
            for l in range(initial_spf,initial_spf+Nspf):
                results_dask=delayed(MLL)(l)
                results_t1_per_Nspf.append(results_dask[0])
                results_t2_per_Nspf.append(results_dask[1])
            results_per_Nspf=dask.compute(*(results_t1_per_Nspf,results_t2_per_Nspf),scheduler='processes',num_workers=NCPU)
            results_t1_per_Nspf=results_per_Nspf[0]
            results_t2_per_Nspf=results_per_Nspf[1]
        elif dask_parallel==False:
            for l in range(initial_spf,initial_spf+Nspf):
                (provi_result_t1,provi_result_t2)=MLL(l)
                results_t1_per_Nspf.append(provi_result_t1)
                results_t2_per_Nspf.append(provi_result_t2)
        # Transpose results_per_Nspf, to get results per walker
        if t1_analysis    == True: results_per_walker_t1=[list(i) for i in zip(*results_t1_per_Nspf)]
        if t2_exploration == True: results_per_walker_t2=[list(i) for i in zip(*results_t2_per_Nspf)]
        # Print final results
        print('--- Final results ---')
        for i in range(adven_per_SPF):
            print('-- Adventurousness: %6.1f --' %(adven[i]))
            if t1_analysis == True:
                print('-- N1 analysis')
                print('- RMSE:',results_per_walker_t1[i][:])
                print('- RMSE Median: %f' %(statistics.median(results_per_walker_t1[i])))
            if t2_exploration == True:
                print('-- N2 exploration')
                print('- [ML_gain_pred, ML_gain_real, error_rel_ML, min_standard, min_ML,ML_gain_real_relative]: %s' %(str(results_per_walker_t2[i])))
                ML_gain_pred          = [item[0] for item in results_per_walker_t2[i]]
                ML_gain_real          = [item[1] for item in results_per_walker_t2[i]]
                error_rel_ML          = [item[2] for item in results_per_walker_t2[i]]
                min_standard          = [item[3] for item in results_per_walker_t2[i]]
                min_ML                = [item[4] for item in results_per_walker_t2[i]]
                ML_gain_real_relative = [item[5] for item in results_per_walker_t2[i]]
                print('- ML_gain_pred Median: %f' %(statistics.median(ML_gain_pred)))
                print('- ML_gain_real Median: %f' %(statistics.median(ML_gain_real)))
                print('- error_rel_ML Median: %f' %(statistics.median(error_rel_ML)))
                print('- min_standard Median: %f' %(statistics.median(min_standard)))
                print('- min_ML Median: %f' %(statistics.median(min_ML)))
                print('- ML_gain_real_relative Median: %f' %(statistics.median(ML_gain_real_relative)))
            print('',flush=True)
        if plot_t1_error_metric == True and error_metric=='rmse':
            plot(error_metric,None,None,None,None,None,None,None,None,results_per_walker_t1)
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
    param = ast.literal_eval(var_value[var_name.index('param')])
    center_min = ast.literal_eval(var_value[var_name.index('center_min')])
    center_max = ast.literal_eval(var_value[var_name.index('center_max')])
    grid_min = ast.literal_eval(var_value[var_name.index('grid_min')])
    grid_max = ast.literal_eval(var_value[var_name.index('grid_max')])
    grid_Delta = ast.literal_eval(var_value[var_name.index('grid_Delta')])
    adven = ast.literal_eval(var_value[var_name.index('adven')])
    t1_time = ast.literal_eval(var_value[var_name.index('t1_time')])
    d_threshold = ast.literal_eval(var_value[var_name.index('d_threshold')])
    t0_time = ast.literal_eval(var_value[var_name.index('t0_time')])
    ML = ast.literal_eval(var_value[var_name.index('ML')])
    error_metric = ast.literal_eval(var_value[var_name.index('error_metric')])
    CV = ast.literal_eval(var_value[var_name.index('CV')])
    k_fold = ast.literal_eval(var_value[var_name.index('k_fold')])
    test_last_proportion = ast.literal_eval(var_value[var_name.index('test_last_proportion')])
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
    KRR_alpha = ast.literal_eval(var_value[var_name.index('KRR_alpha')])
    KRR_kernel = ast.literal_eval(var_value[var_name.index('KRR_kernel')])
    KRR_gamma = ast.literal_eval(var_value[var_name.index('KRR_gamma')])
    optimize_KRR_hyperparams = ast.literal_eval(var_value[var_name.index('optimize_KRR_hyperparams')])
    KRR_alpha_lim = ast.literal_eval(var_value[var_name.index('KRR_alpha_lim')])
    KRR_gamma_lim = ast.literal_eval(var_value[var_name.index('KRR_gamma_lim')])
    t2_time = ast.literal_eval(var_value[var_name.index('t2_time')])
    t2_ML = ast.literal_eval(var_value[var_name.index('t2_ML')])
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
    # Calculate intermediate values 
    width_min=S                                     # Minimum width of each Gaussian function
    width_max=1.0/3.0                               # Maximum width of each Gaussian function
    Amplitude_min=0.0                               # Minimum amplitude of each Gaussian function
    Amplitude_max=1.0                               # Maximum amplitude of each Gaussian function
    N=int(round((1/(S**param))))                    # Number of Gaussian functions of a specific landscape
    adven_per_SPF = len(adven)                      # Number of different adventurousness per SPF
    # Assign variables to a dictionary to check their types and values
    inp_val = {
      "dask_parallel": dask_parallel,
      "NCPU": NCPU,
      "verbosity_level": verbosity_level,
      "log_name": log_name,
      "Nspf": Nspf,
      "initial_spf":initial_spf,
      "S":S,
      "param":param,
      "center_min":center_min,
      "center_max":center_max,
      "grid_min":grid_min,
      "grid_max":grid_max,
      "grid_Delta":grid_Delta,
      "adven":adven,
      "t1_time":t1_time,
      "d_threshold":d_threshold,
      "t0_time":t0_time,
      "ML":ML,
      "error_metric":error_metric,
      "CV":CV,
      "k_fold":k_fold,
      "test_last_proportion":test_last_proportion,
      "n_neighbor":n_neighbor,
      "weights":weights,
      "GBR_criterion":GBR_criterion,
      "GBR_n_estimators":GBR_n_estimators,
      "GBR_learning_rate":GBR_learning_rate,
      "GBR_max_depth":GBR_max_depth,
      "GBR_min_samples_split":GBR_min_samples_split,
      "GBR_min_samples_leaf":GBR_min_samples_leaf,
      "GPR_alpha":GPR_alpha,
      "GPR_length_scale":GPR_length_scale,
      "optimize_GPR_hyperparams":optimize_GPR_hyperparams,
      "GPR_alpha_lim":GPR_alpha_lim,
      "GPR_length_scale_lim":GPR_length_scale_lim,
      "KRR_alpha":KRR_alpha,
      "KRR_kernel":KRR_kernel,
      "KRR_gamma":KRR_gamma,
      "optimize_KRR_hyperparams":optimize_KRR_hyperparams,
      "KRR_alpha_lim":KRR_alpha_lim,
      "KRR_gamma_lim":KRR_gamma_lim,
      "t2_time":t2_time,
      "t2_ML":t2_ML,
      "t2_exploration":t2_exploration,
      "t1_analysis":t1_analysis,
      "diff_popsize":diff_popsize,
      "diff_tol":diff_tol,
      "t2_train_time":t2_train_time,
      "calculate_grid":calculate_grid,
      "plot_t1_exploration":plot_t1_exploration,
      "plot_t1_error_metric":plot_t1_error_metric,
      "plot_contour_map":plot_contour_map,
      "grid_name":grid_name,
    }
    # Check that input values are OK
    check_input_values(inp_val,width_min,width_max,Amplitude_min,Amplitude_max,N,adven_per_SPF)

    return (dask_parallel, NCPU, verbosity_level, log_name, Nspf, 
            S, param, center_min, center_max, grid_min, grid_max, 
            grid_Delta, adven_per_SPF, adven, t1_time, d_threshold, 
            t0_time, ML, error_metric, CV, k_fold, test_last_proportion, 
            n_neighbor, weights, GBR_criterion, GBR_n_estimators, 
            GBR_learning_rate, GBR_max_depth, GBR_min_samples_split, 
            GBR_min_samples_leaf, GPR_alpha, GPR_length_scale, 
            GPR_alpha_lim , GPR_length_scale_lim, KRR_alpha, KRR_kernel, 
            KRR_gamma, optimize_KRR_hyperparams, optimize_GPR_hyperparams, 
            KRR_alpha_lim, KRR_gamma_lim, width_min, width_max, Amplitude_min, 
            Amplitude_max, N, t2_time, t2_ML, t2_exploration, t1_analysis, 
            diff_popsize, diff_tol, t2_train_time, calculate_grid, grid_name,
            plot_t1_exploration,plot_contour_map,plot_t1_error_metric,initial_spf)

# Function to serve as a sanity check of input values
def check_input_values(inp_val,width_min,width_max,Amplitude_min,Amplitude_max,N,adven_per_SPF):
    # Double check boolean variables
    for value in ["calculate_grid","plot_contour_map","dask_parallel","plot_t1_exploration","plot_t1_error_metric","t1_analysis","t2_exploration","optimize_GPR_hyperparams","optimize_KRR_hyperparams"]:
        if type(inp_val[value]) != bool:
            print ('INPUT ERROR: %s should be boolean, but is: %s - %s' %(value, str(inp_val[value]), type(inp_val[value])))
            sys.exit()
    # Double check integer variables
    for value in ["NCPU","Nspf","initial_spf","param","verbosity_level","t0_time","t1_time","t2_time","t2_train_time","k_fold","diff_popsize"]:
        if type(inp_val[value]) != int:
            print ('INPUT ERROR: %s should be integer, but is: %s - %s' %(value, str(inp_val[value]), type(inp_val[value])))
            sys.exit()
    # Double check float variables
    for value in ["S","center_min","center_max","grid_min","grid_max","grid_Delta","d_threshold","test_last_proportion","GPR_alpha","GPR_length_scale","KRR_alpha","KRR_gamma","diff_tol"]:
        if type(inp_val[value]) != float:
            print ('INPUT ERROR: %s should be float, but is: %s - %s' %(value, str(inp_val[value]), type(inp_val[value])))
            sys.exit()
    # Double check string variables
    for value in ["grid_name","log_name","ML","error_metric","t2_ML","CV","weights","GBR_criterion","KRR_kernel"]:
        if type(inp_val[value]) != str:
            print ('INPUT ERROR: %s should be string, but is: %s - %s' %(value, str(inp_val[value]), type(inp_val[value])))
            sys.exit()
    # Double check lists of integers OR lists of floats
    for value in ["adven","GBR_learning_rate"]:
        if type(inp_val[value]) != list:
            print ('INPUT ERROR: %s should be a list, but is: %s - %s' %(value, str(inp_val[value]), type(inp_val[value])))
            sys.exit()
        for term in inp_val[value]:
            if type(term) != int and type(term) != float:
                print ('INPUT ERROR: %s should be a list of integers or reals, but is: %s - %s' %(value, str(inp_val[value]), term, type(term)))
                sys.exit()
    # Double check lists of integers 
    for value in ["n_neighbor","GBR_n_estimators","GBR_max_depth","GBR_min_samples_split","GBR_min_samples_leaf"]:
        if type(inp_val[value]) != list:
            print ('INPUT ERROR: %s should be a list, but is: %s - %s' %(value, str(inp_val[value]), type(inp_val[value])))
            sys.exit()
        for term in inp_val[value]:
            if type(term) != int:
                print ('INPUT ERROR: %s should be a list of integers, but is: %s - %s' %(value, str(inp_val[value]), term, type(term)))
                sys.exit()
    # Double check tuples of integers OR tuples of floats
    for value in ["GPR_alpha_lim","GPR_length_scale_lim","KRR_alpha_lim","KRR_gamma_lim"]:
        if type(inp_val[value]) != tuple:
            print ('INPUT ERROR: %s should be a tuple, but is: %s - %s' %(value, str(inp_val[value]), type(inp_val[value])))
            sys.exit()
        for term in inp_val[value]:
            if type(term) != int and type(term) != float:
                print ('INPUT ERROR: %s should be a tuple of integers or reals, but is: %s - %s' %(value, str(inp_val[value]), term, type(term)))
                sys.exit()
    # Check that specific varialbes are within allowed ranges
    if inp_val["S"] <=0 or inp_val["S"] >= width_max:
        print ('INPUT ERROR: S should be in range (%f,%f), but it is: %f' %(0.0,width_max,S))
        sys.exit()
    allowed_verbosity_level = [0,1,2]
    if inp_val["verbosity_level"] not in allowed_verbosity_level:
        print ('INPUT ERROR: verbosity_level should be in %s, but it is: %i' %(str(allowed_verbosity_level),verbosity_level))
        sys.exit()
    allowed_ML=[None,'kNN','GBR','GPR','KRR']
    if inp_val["ML"] not in allowed_ML:
        print ('INPUT ERROR: ML should be in %s, but it is: %s' %(str(allowed_ML),ML))
        sys.exit()
    allowed_error_metric=[None,'rmse']
    if inp_val["error_metric"] not in allowed_error_metric:
        print ('INPUT ERROR: error_metric should be in %s, but it is: %s' %(str(allowed_error_metric),error_metric))
        sys.exit()
    allowed_t2_ML = [None,'kNN','GPR','KRR']
    if inp_val["t2_ML"] not in allowed_t2_ML:
        print ('INPUT ERROR: t2_ML should be in %s, but it is: %s' %(str(allowed_t2_ML),t2_ML))
        sys.exit()
    allowed_CV=['kf','loo','time-sorted']
    if inp_val["CV"] not in allowed_CV:
        print ('INPUT ERROR: CV should be in %s, but it is: %s' %(str(allowed_CV),CV))
        sys.exit()
    # Print read input values to standard output
    print('# Printing input values:')
    for var in inp_val:
        print(var, "=", inp_val[var])
    print('# Intermediate values:')
    print('width_min =',width_min)
    print('width_max =',width_max)
    print('Amplitude_min =',Amplitude_min)
    print('Amplitude_max =',Amplitude_max)
    print('N =',N)
    print('adven_per_SPF =',adven_per_SPF)
    print('########################')
    print('### INPUT CHECKED OK ###')
    print('########################')

### Function to calculate SPF grid
def generate_grid(l):
    time_taken1 = time()-start
    Amplitude      = []
    center_N       = [[] for i in range(N)]
    width_N        = [[] for i in range(N)]
    dim_list       = [[] for i in range(param)]
    G_list         = []
    f_out = open('%s_%s.log' % (grid_name,l), 'w')
    if verbosity_level>=1: 
        f_out.write("################################## \n")
        f_out.write('## Start: "generate_grid" function \n')
        f_out.write("################################## \n")
        f_out.write("########################### \n")
        f_out.write("###### Landscape %i ####### \n" % (l))
        f_out.write("########################### \n")
        f_out.write("%s %6.2f \n" % ('Verbosity level:', verbosity_level))
        f_out.flush()
    # Assign Gaussian values
    for i in range(N):
        random.seed(a=None)
        Amplitude.append(random.uniform(Amplitude_min,Amplitude_max))
        random.seed(a=None)
        am_i_negative=random.randint(0,1)
        if am_i_negative==0: Amplitude[i]=-Amplitude[i]
        for dim in range(param):
            random.seed(a=None)
            center_N[i].append(random.uniform(center_min,center_max))
            random.seed(a=None)
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
    # Calculate G grid
    counter=0
    if verbosity_level>=1: f_out.write("%8s %11s %15s \n" % ("i","x","G"))
    Nlen=(int(round((grid_max-grid_min)/(grid_Delta)+1)))
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

    Ngrid=int(round((grid_max/grid_Delta+1)**param))   # calculate number of grid points
    max_G=max(G_list)
    min_G=min(G_list)
    if verbosity_level>=1:
        max_G_index=int(round(np.where(G_list == np.max(G_list))[0][0]))
        min_G_index=int(round(np.where(G_list == np.min(G_list))[0][0]))
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
        plot('contour',l,None,dim_list,G_list,None,None,None,None,None)
        #plot('3d_landscape',l,None,dim_list,G_list,None,None,None,None,None)
    #return dim_list, G_list, Ngrid, max_G # not needed as we write grid to file now
    time_taken2 = time()-start
    f_out.write("Generate grid took %0.4f seconds\n" %(time_taken2-time_taken1))
    return None

### Function doing most of the heavy lifting
def MLL(l):
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
    #dim_list, G_list, Ngrid, max_G = generate_grid(l,f_out)
    #  New: read grid from file
    filename = grid_name + '_' + str(l) + '.log'
    dim_list       = [[] for i in range(param)]
    G_list         = []
    with open(filename) as file_in:
        for line in file_in:
            counter_word=-1
            # read only lines that start with an integer (i). The rest are ignored
            try:
                if isinstance(int(line.split()[0]),int):
                    for word in line.split():
                        if counter_word >= 0 and counter_word < param:  dim_list[counter_word].append(float(word))
                        if counter_word == param: G_list.append(float(word))
                        counter_word=counter_word+1
            except:
                pass
    Ngrid=int(round((grid_max/grid_Delta+1)**param))   # calculate number of grid points
    max_G=max(G_list)
    min_G=min(G_list)
    ##################################
    time_taken2 = time()-start
    if verbosity_level>=1:
        f_out.write("Generate grid took %0.4f seconds\n" %(time_taken2-time_taken1))
    # For each walker
    for w in range(adven_per_SPF):
        # Step 1) Perform t1 exploration
        time_taken1 = time()-start
        X0,y0,unique_t0 = explore_landscape(l,w,dim_list,G_list,f_out,Ngrid,max_G,t0_time,0,0,None,None,False,None,None)
        X1,y1,unique_t1 = explore_landscape(l,w,dim_list,G_list,f_out,Ngrid,max_G,0,t1_time,0,None,None,False,X0,y0)
        time_taken2 = time()-start
        if verbosity_level>=1:
            f_out.write("N1 exploration took %0.4f seconds\n" %(time_taken2-time_taken1))
        if t1_analysis == True:
        # Step 2A) Calculate error_metric
            time_taken1 = time()-start
            if ML=='kNN': error_metric_result=kNN(X1,y1,l,w,f_out,None,None,1,None)
            if ML=='GBR': error_metric_result=GBR(X1,y1,l,w,f_out)
            if ML=='GPR':
                hyperparams=[GPR_alpha,GPR_length_scale]
                if optimize_GPR_hyperparams == False:
                    error_metric_result=GPR(hyperparams,X1,y1,l,w,f_out,None,None,1,None,False)
                else:
                    mini_args=(X1,y1,l,w,f_out,None,None,1,None,True)
                    bounds = [GPR_alpha_lim]+[GPR_length_scale_lim]
                    solver=differential_evolution(GPR,bounds,args=mini_args,popsize=diff_popsize,tol=diff_tol)
                    best_hyperparams = solver.x
                    best_rmse = solver.fun
                    if verbosity_level>=1: 
                        f_out.write("Best hyperparameters: %s . Best rmse: %f \n" %(str(best_hyperparams),best_rmse))
                        f_out.flush()
                    error_metric_result = best_rmse
                    if CV=='time-sorted':
                        error_metric_result=GPR(best_hyperparams,X1,y1,l,w,f_out,None,None,1,None,False)
            if ML=='KRR':
                hyperparams=[KRR_alpha,KRR_gamma]
                if optimize_KRR_hyperparams == False:
                    error_metric_result=KRR(hyperparams,X1,y1,l,w,f_out,None,None,1,None,False)
                else:
                    mini_args=(X1,y1,l,w,f_out,None,None,1,None,True)
                    bounds = [KRR_alpha_lim]+[KRR_gamma_lim]
                    solver=differential_evolution(KRR,bounds,args=mini_args,popsize=diff_popsize,tol=diff_tol)
                    best_hyperparams = solver.x
                    best_rmse = solver.fun
                    if verbosity_level>=1: 
                        f_out.write("Best hyperparameters: %s . Best rmse: %f \n" %(str(best_hyperparams),best_rmse))
                        f_out.flush()
                    error_metric_result = best_rmse
                    if CV=='time-sorted':
                        error_metric_result=KRR(best_hyperparams,X1,y1,l,w,f_out,None,None,1,None,False)
            error_metric_list.append(error_metric_result)
            result1 = error_metric_list
            time_taken2 = time()-start
            # Plot 2d exploration
            if plot_t1_exploration == True and param == 2:
                plot('t1_exploration',l,w,dim_list,G_list,X0,y0,X1,y1,None)
            if verbosity_level>=1:
                f_out.write("N1 analysis took %0.4f seconds\n" %(time_taken2-time_taken1))
        # Step 2B) Perform t2 exploration
        if t2_exploration == True:
            # Step 2.B.1) Perform t2 exploration with weighted random explorer
            time_taken1 = time()-start
            X2a,y2a,unique_t2a = explore_landscape(l,w,dim_list,G_list,f_out,Ngrid,max_G,0,unique_t1,t2_time,X1,y1,False,None,None)
            if verbosity_level>=1:
                time_taken2 = time()-start
                f_out.write("t2 standard exploration took %0.4f seconds\n" %(time_taken2-time_taken1))
            # Step 2.B.2) Perform t2 exploration with ML-guided explorer
            time_taken1 = time()-start
            X2b,y2b,unique_t2b = explore_landscape(l,w,dim_list,G_list,f_out,Ngrid,max_G,0,unique_t1,t2_time,X1,y1,True,None,None)
            if verbosity_level>=1:
                time_taken2 = time()-start
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
            # calculate MLgain with respect to minimum value obtained with a standard exploration, whether in t1 or t2
            if min(y2a) < min(y1):
                min_standard = min(y2a)
            else:
                min_standard = min(y1)
            MLgain_pred = min_standard - min(y2b)
            MLgain_real = min_standard -  y_real 
            MLgain_real_relative = (min_standard -  y_real )/abs(min_standard)
            # Print t2 exploration results
            if verbosity_level>=1: 
                f_out.write("#####################\n")
                f_out.write("## SUMMARY RESULTS ##\n")
                f_out.write("#####################\n")
                f_out.write("## Initial random exploration: \n")
                f_out.write("N0 = %i \n" %(t0_time))
                f_out.write("################ \n")
                f_out.write("## N1 exploration: \n")
                f_out.write("N1 = %i \n" %(t1_time))
                f_out.write("Last value: X1 index (unique timestep): %i\n" %(len(y1)-1))
                f_out.write("Last value: X1: %s, y: %s\n" %(str(X1[-1]),str(y1[-1])))
                f_out.write("Minimum value: X1 index (unique timestep): %s\n" %(str(np.where(y1 == np.min(y1))[0][0])))
                f_out.write("Minimum value: X1: %s, y: %s\n" %(str(X1[np.where(y1 == np.min(y1))][0]),str(min(y1))))
                f_out.write("################ \n")
                f_out.write("## N2 standard exploration: \n")
                f_out.write("N2 = %i \n" %(t2_time))
                f_out.write("Last value: X2a index (unique timestep): %i\n" %(len(y2a)-1))
                f_out.write("Last value: X2a: %s, y2a: %s\n" %(str(X2a[-1]),str(y2a[-1])))
                f_out.write("Minimum value: X2a index (unique timestep): %s\n" %(str(np.where(y2a == np.min(y2a))[0][0])))
                f_out.write("Minimum value: X2a: %s, y2a: %s\n" %(str(X2a[np.where(y2a == np.min(y2a))][0]),str(min(y2a))))
                f_out.write("################ \n")
                f_out.write("## N2 ML exploration: \n")
                f_out.write("N2 = %i \n" %(t2_time))
                f_out.write("Last value: X2b index (unique timestep): %i\n" %(len(y2b)-1))
                f_out.write("Last value: X2b: %s, y2b: %s\n" %(str(X2b[-1]),str(y2b[-1])))
                f_out.write("Minimum value: X2b index (unique timestep): %s\n" %(str(np.where(y2b == np.min(y2b))[0][0])))
                f_out.write("Minimum predicted value: X2b: %s, y2b: %s\n" %(str(X2b[np.where(y2b == np.min(y2b))][0]),str(min(y2b))))
                f_out.write("Minimum real value: X2b: %s, y2b: %s\n" %(str(x_real),str(y_real)))
    
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
    if verbosity_level>=1:
        time_taken2 = time()-start
        f_out.write("Rest of MLL took %0.4f seconds\n" %(time_taken2-time_taken1))
        f_out.write("I am returning these values: %s, %s\n" %(str(result1), str(result2)))
        f_out.close()
    return (result1, result2)

# Function that explores the SPF map to generate research landscapes
def explore_landscape(l,w,dim_list,G_list,f_out,Ngrid,max_G,t0,t1,t2,Xi,yi,ML_explore,X0,y0):
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
    if verbosity_level>=2: 
        f_out.write("###################################### \n")
        f_out.write('## Start: "explore_landscape" function \n')
        f_out.write("###################################### \n")
        f_out.write("Adventurousness: %f \n" % (adven[w]))
        f_out.write("Number of points per dimension: %i \n" %Nx)
        f_out.write("###################################### \n")
        f_out.flush()
    ### Perform t0 exploration ###
    # could be simplified by just chosing a random number from 0 to Ngrid-1 per N0 step
    if verbosity_level>=1 and t0 !=0: 
        f_out.write("################################## \n")
        f_out.write("## Start initial random exploration\n" %())
        f_out.write("# New Adventurousness: %.2f \n" % (adven[w]))
        f_out.write("# Number of grid points per dimension: %i \n" %Nx)
        f_out.write("################################## \n")
        f_out.flush()
    if t0 != 0:
        for t in range(t0):
            for i in range(param):
                random.seed(a=None)
                num=int(round(random.randint(0,Ngrid-1)))
                if t==0:
                    walker_x.append(dim_list[i][num])
                else:
                    walker_x[i]=dim_list[i][num]
            for i in range(param):
                path_x[i].append(walker_x[i])
            num_in_grid=0.0
            for i in range(param):
                num_in_grid=num_in_grid + round(walker_x[param-1-i]*(Nx-1))*(Nx**i) # Needed to round product to avoid propagation of small error (relevant for larger dimensionality)
            num_in_grid=int(round(num_in_grid))
            path_G.append(G_list[num_in_grid])
            list_t.append(t)
    
            if verbosity_level>=1:
                line = []
                for j in range(param):
                    line.append((walker_x[j]))
                line.append((G_list[num_in_grid]))
                f_out.write("timestep %6i %2s %s\n" % (t,"",str(line)))
                f_out.flush()
        if t1 == 0:
            X = path_x
            y = path_G
            return X,y,len(y)
    # Set values for t1 exploration
    if t2==0:
        path_x = copy.deepcopy(X0)
        path_G = copy.deepcopy(y0)
        t_ini=t0_time
        t_fin=t1+t0_time
        for i in range(param): # set walker_x to last path_x
            walker_x.append(path_x[i][-1])
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
        if verbosity_level>=1: 
            f_out.write("############################## \n")
            f_out.write("## Start a-weighted exploration\n" %())
            f_out.write("############################## \n")
            f_out.flush()
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
            random.seed(a=None)
            draw=random.choice(range(special_points))
            draw_in_grid=G_list.index(minimum_path_G[draw])
            draw_in_grid_list=[i for i, e in enumerate(G_list) if e == minimum_path_G[draw] ]
            #####################################################################################
            # Added fix to select correct draw_in_grid value from list, when several points in grid have same G value
            if len(draw_in_grid_list) > 1:
                for i in range(len(draw_in_grid_list)):
                    counter_select_from_list=0
                    for j in range(param):
                        if dim_list[j][draw_in_grid_list[i]] == minimum_path_x[j][draw]:
                            counter_select_from_list = counter_select_from_list+1
                    if counter_select_from_list == param:
                        draw_in_grid = draw_in_grid_list[i]
            #####################################################################################
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
            # Check for inconsistencies
            for k in range(len(draw_in_grid_list)):
                counter_param=0
                for i in range(param):
                    if minimum_path_x[i][draw] == dim_list[i][draw_in_grid_list[k]]:
                        counter_param=counter_param+1
                if counter_param==param:
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
                        index[j]=int(round(index[j]))
                    index.append(draw_in_grid-subs)
                    index[param]=int(round(index[param]))
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
                print("STOP - ERROR: No new candidate points found within threshold")
                print("STOP - ERROR: At Nspf:", l, ". Walker:", w, ". Time:", t)
                f_out.write("STOP - ERROR: No new candidate points found within threshold\n")
                sys.exit()

            if len(range((P*2+1)**param)) != len(prob):
                print("STOP - ERROR: Problem with number of nearby points considered for next step")
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
                if t2==0:
                    f_out.write("timestep %6i %2s %s\n" % (t,"",str(line)))
                else:
                    f_out.write("timestep-N2 a-weighted %6i %2s %s\n" % (t,"",str(line)))
                f_out.flush()
            # update x_param and y with new values
            for i in range(param):
                x_param[i].append(walker_x[i])
            y.append(neighbor_G[draw])
        # calculate final X and y
        X,y = create_X_and_y(f_out,x_param,y)
    ### Perform t2 ML exploration
    else:
        if verbosity_level>=1: 
            f_out.write("############################# \n")
            f_out.write("## Start ML-guided exploration\n" %())
            f_out.write("############################# \n")
            f_out.flush()
        for t in range(t_ini,t_fin):
            time_taken0 = time()-start
            time_taken1 = time()-start
            # For t2=0, calculate bubble for all points previously visited (slow process)
            if t==0:
                # initialize values
                x_bubble=[[] for j in range(param)]
                y_bubble=[]
                # For each point in Xi
                for k in range(len(path_G)):
                    counter3=0
                    del prob[:]
                    del neighbor_walker[:][:]
                    del neighbor_G[:]
                    prob_sum=0.0
                    # get coordinates of kth point in SPF grid
                    new_k_in_grid=[[] for j in range(param)]
                    # for each parameter
                    k_list=[]
                    for j in range(param):
                        # calculate list of indeces in grid that match path_x values
                        k_list = [i for i, x in enumerate(dim_list[j]) if x == path_x[j][k]]
                        new_k_in_grid[j].append(k_list)
                        if verbosity_level>=2: f_out.write("Looking for path_x[j][k]: %f\n" %(path_x[j][k]))
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
                    #######################
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
                            line.append((path_x[j][k]))
                        line.append((path_G[k]))
                        f_out.write("Check around point: %s\n" %(str(line)))
                        f_out.write("%6s %20s %11s %13s \n" % ("i","[x, G]","Prob","distance"))
                        f_out.flush()
                    ## Index_i serves to explore neighbors: from -P+1 to P for each parameter
                    for index_i in itertools.product(range(-P+1,P), repeat=param):
                        try: # Use try/except to ignore errors when looking for points outside of landscape
                            index=[]
                            subs=0.0
                            # Assign indeces of grid corresponding to wanted points
                            for j in range(param):
                                index.append(k_in_grid - Nx**(param-1-j)*index_i[j])
                                subs=subs+Nx**(param-1-j)*index_i[j]
                                index[j]=int(round(index[j]))
                            index.append(k_in_grid-subs)
                            index[param]=int(round(index[param]))
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
                if verbosity_level>=1: 
                    f_out.write("value of k_in_grid is: %i\n" %(k_in_grid))
                    f_out.flush()
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
                            index[j]=int(round(index[j]))
                        index.append(k_in_grid-subs)
                        index[param]=int(round(index[param]))
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
            if t2_ML=='kNN': min_point=kNN(x_bubble,y_bubble,l,w,f_out,path_x,path_G,2,t)
            if t2_ML=='GPR':
                hyperparams=[GPR_alpha,GPR_length_scale]
                if optimize_GPR_hyperparams == False:
                    min_point=GPR(hyperparams,x_bubble,y_bubble,l,w,f_out,path_x,path_G,2,t,False)
                else:
                    mini_args=(path_x,path_G,l,w,f_out,None,None,1,t,False) # get rmse fitting previous points
                    bounds = [GPR_alpha_lim]+[GPR_length_scale_lim]
                    solver=differential_evolution(GPR,bounds,args=mini_args,popsize=diff_popsize,tol=diff_tol)
                    best_hyperparams = solver.x
                    best_rmse = solver.fun
                    if verbosity_level>=1: 
                        f_out.write("Best hyperparameters: %s . Best rmse: %f \n" %(str(best_hyperparams),best_rmse))
                        f_out.flush()
                    hyperparams=[best_hyperparams[0],best_hyperparams[1]]
                    min_point=GPR(hyperparams,x_bubble,y_bubble,l,w,f_out,path_x,path_G,2,t,False)
            if t2_ML=='KRR':
                hyperparams=[KRR_alpha,KRR_gamma]
                if optimize_KRR_hyperparams == False:
                    min_point=KRR(hyperparams,x_bubble,y_bubble,l,w,f_out,path_x,path_G,2,t,False)
                else:
                    mini_args=(path_x,path_G,l,w,f_out,None,None,1,t,False) # get rmse fitting previous points
                    bounds = [KRR_alpha_lim]+[KRR_gamma_lim]
                    solver=differential_evolution(KRR,bounds,args=mini_args,popsize=diff_popsize,tol=diff_tol)
                    best_hyperparams = solver.x
                    best_rmse = solver.fun
                    if verbosity_level>=1: 
                        f_out.write("Best hyperparameters: %s . Best rmse: %f \n" %(str(best_hyperparams),best_rmse))
                        f_out.flush()
                    hyperparams=[best_hyperparams[0],best_hyperparams[1]]
                    min_point=KRR(hyperparams,x_bubble,y_bubble,l,w,f_out,path_x,path_G,2,t,False)
            # print x_bubble corresponding to min(y_bubble)
            if verbosity_level>=1: 
                #f_out.write("## At time %i, minimum predicted point is: %s\n" %(t, str(min_point)))
                f_out.write("timestep-N2 ML-guided %i: %s\n" %(t, str(min_point)))
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
        # get final X,y
        X,y = create_X_and_y(f_out,x_param,y)
    return X,y,len(y)

# Function to transform X and y to standard format
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
        f_out.write("# Unique points: %i\n" % (len(y)))
        if len(y) < 20:
            for i in range(len(y)):
                f_out.write("%s %f\n" %(str(X[i]),y[i]))
        else:
            for i in range(10):
                f_out.write("%s %f\n" %(str(X[i]),y[i]))
            f_out.write("    [...]  \n")
            for i in range(len(y)-10,len(y)):
                f_out.write("%s %f\n" %(str(X[i]),y[i]))
        f_out.flush()
    return X,y

# Function to do kNN regression
def kNN(X,y,l,w,f_out,Xtr,ytr,mode,t):
    # initialize values
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
            if verbosity_level>=2: 
                f_out.write('## Start: "kNN" function \n')
                f_out.write('-------- \n')
                f_out.write('Perform k-NN \n')
                f_out.write('k= %i \n' % (n_neighbor[n]))
                f_out.write('cross_validation %i - fold \n' % (k_fold))
                f_out.write('weights %s \n' % (weights))
                f_out.write('-------- \n')
                f_out.flush()
            # assign splits to kf or and loo
            if CV=='kf':
                kf = KFold(n_splits=k_fold,shuffle=True,random_state=None)
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
                    y_pred = knn.fit(X_train_scaled, y_train).predict(X_test_scaled)
                    # add y_test and y_pred values to general real_y and predicted_y
                    for i in range(len(y_test)):
                        real_y.append(y_test[i])
                        predicted_y.append(y_pred[i])
                    # if large verbosity, calculate r and rmse at each split. Then print extra info
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
            elif CV=='time-sorted':
                #######################################################################
                # Use (1-'test_last_proportion') as training, and 'test_last_proportion' as test data
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_last_proportion,random_state=None,shuffle=False)
                # scale data
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                # fit kNN with (X_train, y_train), and predict X_test
                knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbor[n], weights=weights)
                kf = KFold(n_splits=k_fold,shuffle=True,random_state=None)
                validation=kf.split(X_train_scaled)
                y_pred_final = []
                y_valid_final  = []
                for train_index, valid_index in validation:
                    X_new_train,X_new_valid=X_train_scaled[train_index],X_train_scaled[valid_index]
                    y_new_train,y_new_valid=y_train[train_index],y_train[valid_index]
                    # fit GPR with (X_new_train, y_new_train), and predict X_new_valid
                    y_pred = knn.fit(X_new_train, y_new_train).predict(X_new_valid)
                    for i in range(len(y_new_valid)):
                        y_valid_final.append(y_new_valid[i])
                        y_pred_final.append(y_pred[i])
                # calculate final r and rmse
                total_r_pearson,_=pearsonr(y_valid_final,y_pred_final)
                mse = mean_squared_error(y_valid_final,y_pred_final)
                total_rmse = np.sqrt(mse)
                # print verbose info
                if verbosity_level>=2:
                    f_out.write("Train with first %i points \n" % (len(X_train)))
                    f_out.write("%s \n" % (str(X_train)))
                    f_out.write("Test with last %i points \n" % (len(X_test)))
                    f_out.write("%s \n" % (str(X_test)))
                    f_out.write('Landscape %i . Adventurousness: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],total_r_pearson,total_rmse))
            # Print last verbose info for kNN
            if verbosity_level>=2:
                f_out.write('Final r_pearson, rmse: %f, %f \n' % (total_r_pearson,total_rmse))
                f_out.flush()
            if n==0 or total_rmse<prev_total_rmse:
                if error_metric=='rmse': result=total_rmse
                final_k = n_neighbor[n]
                prev_total_rmse = total_rmse
             #######################################################################
        # After k is optimized, predict X-test
        knn = neighbors.KNeighborsRegressor(n_neighbors=final_k, weights=weights)
        y_pred = knn.fit(X_train_scaled, y_train).predict(X_test_scaled)
        # calculate final r and rmse
        total_r_pearson,_=pearsonr(y_test,y_pred)
        mse = mean_squared_error(y_test, y_pred)
        total_rmse = np.sqrt(mse)
        result=total_rmse
        #######################################################################
        f_out.write("---------- \n")
        f_out.write("Final Optimum k: %i, rmse: %f \n" % (final_k,result))
    elif mode==2:
        for n in range(len(n_neighbor)):
            # initialize values
            real_y=[]
            predicted_y=[]
            provi_result = []
            # verbose info
            if verbosity_level>=2:
                f_out.write('## Start: "kNN" function \n')
                f_out.write('-------- \n')
                f_out.write('Perform k-NN \n')
                f_out.write('k= %i \n' % (n_neighbor[n]))
                f_out.write('weights %s \n' % (weights))
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
                if verbosity_level>=1: f_out.write("# At time N2=%i, I am training new model\n" %(t))
                knn.fit(X_train_scaled, y_train)
                dump(knn, open('knn_%i.pkl' %(l), 'wb'))
                time_taken2 = time()-start
            else:
                if verbosity_level>=1: f_out.write("# At time N2=%i, I am reading previous trained model\n" %(t))
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

# Function to do GBR regression
def GBR(X,y,l,w,f_out):
    prev_total_rmse = 0
    gbr_counter = 0
    for gbr1 in range(len(GBR_n_estimators)):
        for gbr2 in range(len(GBR_learning_rate)):
            for gbr3 in range(len(GBR_max_depth)):
                for gbr4 in range(len(GBR_min_samples_split)):
                    for gbr5 in range(len(GBR_min_samples_leaf)):
                        if verbosity_level>=2: 
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
                    
                        kf = KFold(n_splits=k_fold,shuffle=True,random_state=None)
                        average_r=0.0
                        average_r_pearson=0.0
                        average_rmse=0.0
                        if CV=='kf':
                            kf = KFold(n_splits=k_fold,shuffle=True,random_state=None)
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
                                # scale data
                                scaler = preprocessing.StandardScaler().fit(X_train)
                                X_train_scaled = scaler.transform(X_train)
                                X_test_scaled = scaler.transform(X_test)
                                GBR = GradientBoostingRegressor(criterion=GBR_criterion,n_estimators=GBR_n_estimators[gbr1],learning_rate=GBR_learning_rate[gbr2],max_depth=GBR_max_depth[gbr3],min_samples_split=GBR_min_samples_split[gbr4],min_samples_leaf=GBR_min_samples_leaf[gbr5])
                                y_pred = GBR.fit(X_train_scaled, y_train).predict(X_test_scaled)
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
                        elif CV=='time-sorted':
                            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_last_proportion,random_state=None,shuffle=False)
                            # scale data
                            scaler = preprocessing.StandardScaler().fit(X_train)
                            X_train_scaled = scaler.transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            GBR = GradientBoostingRegressor(criterion=GBR_criterion,n_estimators=GBR_n_estimators[gbr1],learning_rate=GBR_learning_rate[gbr2],max_depth=GBR_max_depth[gbr3],min_samples_split=GBR_min_samples_split[gbr4],min_samples_leaf=GBR_min_samples_leaf[gbr5])
                            kf = KFold(n_splits=k_fold,shuffle=True,random_state=None)
                            validation=kf.split(X_train_scaled)
                            y_pred_final = [] 
                            y_valid_final  = [] 
                            for train_index, valid_index in validation:
                                X_new_train,X_new_valid=X_train_scaled[train_index],X_train_scaled[valid_index]
                                y_new_train,y_new_valid=y_train[train_index],y_train[valid_index]
                                # fit GBR with (X_new_train, y_new_train), and predict X_new_valid
                                y_pred = GBR.fit(X_new_train, y_new_train).predict(X_new_valid)
                                for i in range(len(y_new_valid)):
                                    y_valid_final.append(y_new_valid[i])
                                    y_pred_final.append(y_pred[i])
                            # calculate final r and rmse
                            total_r_pearson,_=pearsonr(y_valid_final,y_pred_final)
                            mse = mean_squared_error(y_valid_final,y_pred_final)
                            total_rmse = np.sqrt(mse)
                            # print verbose info
                            if verbosity_level>=2: 
                                f_out.write("Train with first %i points \n" % (len(X_train)))
                                f_out.write("%s \n" % (str(X_train)))
                                f_out.write("Test with last %i points \n" % (len(X_test)))
                                f_out.write("%s \n" % (str(X_test)))
                                f_out.write('Landscape %i . Adventurousness: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],total_r_pearson,total_rmse))
                        # Print last verbose info for GBR
                        if verbosity_level>=2: 
                            f_out.write('Final r_pearson, rmse: %f, %f \n' % (total_r_pearson,total_rmse))
                            f_out.flush()
                        if gbr_counter==0 or total_rmse<prev_total_rmse:
                            final_gbr = []
                            if error_metric=='rmse': result=total_rmse
                            final_gbr = [GBR_n_estimators[gbr1],GBR_learning_rate[gbr2],GBR_max_depth[gbr3],GBR_min_samples_split[gbr4],GBR_min_samples_leaf[gbr5]]
                            f_out.write("New final_gbr: %s \n" % (str(final_gbr)))
                            prev_total_rmse = total_rmse
                        gbr_counter=gbr_counter+1
    # Print optimized hyperparams
    f_out.write("---------- \n")
    f_out.write("Final Optimum GBR: %s, rmse: %f \n" % (str(final_gbr),result))
    # After hyperparams are optimized, predict X_test
    GBR = GradientBoostingRegressor(criterion=GBR_criterion,n_estimators=final_gbr[0],learning_rate=final_gbr[1],max_depth=final_gbr[2],min_samples_split=final_gbr[3],min_samples_leaf=final_gbr[4])
    y_pred = GBR.fit(X_train_scaled, y_train).predict(X_test_scaled)
    # calculate final r and rmse
    total_r_pearson,_=pearsonr(y_test,y_pred)
    mse = mean_squared_error(y_test, y_pred)
    total_rmse = np.sqrt(mse)
    result=total_rmse
    return result

# CALCULATE GPR #
def GPR(hyperparams,X,y,l,w,f_out,Xtr,ytr,mode,t,opt_train):
    # assign hyperparameters
    GPR_alpha,GPR_length_scale = hyperparams
    # initialize values
    average_r=0.0
    average_r_pearson=0.0
    average_rmse=0.0
    real_y=[]
    predicted_y=[]
    counter_split=0
    # CASE1: Calculate error metric
    if mode==1:
        # verbose info
        if verbosity_level>=2: 
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
            kf = KFold(n_splits=k_fold,shuffle=True,random_state=None)
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
                # if large verbosity, calculate r and rmse at each split. Then print extra info
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
        elif CV=='time-sorted':
            if opt_train==True:
                # Use (1-'test_last_proportion') as training, and 'test_last_proportion' as test data
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_last_proportion,random_state=None,shuffle=False)
                # scale data
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                kernel = RBF(length_scale=GPR_length_scale, length_scale_bounds=(1e-3, 1e+3))
                GPR = GaussianProcessRegressor(kernel=kernel, alpha=GPR_alpha, optimizer=None, normalize_y=False, copy_X_train=True, random_state=None)
                #######################################################
                # Separate between new_train and new_valid
                kf = KFold(n_splits=k_fold,shuffle=True,random_state=None)
                validation=kf.split(X_train_scaled)
                y_pred_final = []
                y_valid_final  = []
                for train_index, valid_index in validation:
                    X_new_train,X_new_valid=X_train_scaled[train_index],X_train_scaled[valid_index]
                    y_new_train,y_new_valid=y_train[train_index],y_train[valid_index]
                    #######################################################
                    # fit GPR with (X_new_train, y_new_train), and predict X_new_valid
                    y_pred = GPR.fit(X_new_train, y_new_train).predict(X_new_valid)
                    for i in range(len(y_new_valid)):
                        y_valid_final.append(y_new_valid[i])
                        y_pred_final.append(y_pred[i])
                # calculate final r and rmse
                total_r_pearson,_=pearsonr(y_valid_final,y_pred_final)
                mse = mean_squared_error(y_valid_final,y_pred_final)
                total_rmse = np.sqrt(mse)
                # print verbose info
                if  verbosity_level>=2: 
                    f_out.write("Train with first %i points \n" % (len(X_train)))
                    f_out.write("%s \n" % (str(X_train)))
                    f_out.write("Test with last %i points \n" % (len(X_test)))
                    f_out.write("%s \n" % (str(X_test)))
                    f_out.write('Landscape %i . Adventurousness: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],total_r_pearson,total_rmse))
            if opt_train==False:
                # Test last-10% with hyperparams optimized in first 90%
                # Use (1-'test_last_proportion') as training, and 'test_last_proportion' as test data
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_last_proportion,random_state=None,shuffle=False)
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
        # Print last verbose info for GPR
        if verbosity_level>=2: 
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
        if verbosity_level>=2:
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
            f_out.write("# At time N2=%i, I am training new model\n" %(t))
            GPR.fit(X_train_scaled, y_train)
            #dump(KRR, open('KRR.pkl', 'wb'))
            if t2_train_time !=1: dump(GPR, open('GPR_%i.pkl' %(l), 'wb'))
        else:
            f_out.write("# At time N2=%i, I am reading previous trained model\n" %(t))
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
            f_out.flush()
        # add predicted value to result
        for j in range(param):
            result.append(X_test[min_index][j])
        result.append(min(predicted_y))
    return result

# Function to do kernel ridge regression (KRR)
def KRR(hyperparams,X,y,l,w,f_out,Xtr,ytr,mode,t,opt_train):
    # assign hyperparameters
    KRR_alpha,KRR_gamma = hyperparams
    # initialize values
    average_r=0.0
    average_r_pearson=0.0
    average_rmse=0.0
    real_y=[]
    predicted_y=[]
    counter_split=0
    # CASE1: Calculate error metric
    if mode==1:
        # verbose info
        if verbosity_level>=2: 
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
            kf = KFold(n_splits=k_fold,shuffle=True,random_state=None)
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
        elif CV=='time-sorted':
            if opt_train==True:
                # Use (1-'test_last_proportion') as training, and 'test_last_proportion' as test data
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_last_proportion,random_state=None,shuffle=False)
                # scale data
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                KRR = KernelRidge(alpha=KRR_alpha,kernel=KRR_kernel,gamma=KRR_gamma)
                #######################################################
                # Separate between new_train and new_valid
                kf = KFold(n_splits=k_fold,shuffle=True,random_state=None)
                validation=kf.split(X_train_scaled)
                y_pred_final = []
                y_valid_final  = []
                for train_index, valid_index in validation:
                    X_new_train,X_new_valid=X_train_scaled[train_index],X_train_scaled[valid_index]
                    y_new_train,y_new_valid=y_train[train_index],y_train[valid_index]
                    #######################################################
                    # fit KRR with (X_new_train, y_new_train), and predict X_new_valid
                    y_pred = KRR.fit(X_new_train, y_new_train).predict(X_new_valid)
                    for i in range(len(y_new_valid)):
                        y_valid_final.append(y_new_valid[i])
                        y_pred_final.append(y_pred[i])
                # calculate final r and rmse
                total_r_pearson,_=pearsonr(y_valid_final,y_pred_final)
                mse = mean_squared_error(y_valid_final,y_pred_final)
                total_rmse = np.sqrt(mse)
                # print verbose info
                if  verbosity_level>=2: 
                    f_out.write("Train with first %i points \n" % (len(X_train)))
                    f_out.write("%s \n" % (str(X_train)))
                    f_out.write("Test with last %i points \n" % (len(X_test)))
                    f_out.write("%s \n" % (str(X_test)))
                    f_out.write('Landscape %i . Adventurousness: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],total_r_pearson,total_rmse))
            if opt_train==False:
                # Use (1-'test_last_proportion') as training, and 'test_last_proportion' as test data
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_last_proportion,random_state=None,shuffle=False)
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
        # Print last verbose info for KRR
        if verbosity_level>=2: 
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
        if verbosity_level>=2:
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
            f_out.write("# At time N2=%i, I am training new model\n" %(t))
            KRR.fit(X_train_scaled, y_train)
            if t2_train_time !=1: dump(KRR, open('KRR_%i.pkl' %(l), 'wb'))
        else:
            f_out.write("# At time N2=%i, I am reading previous trained model\n" %(t))
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

# Function to do different plots
def plot(flag,l,w,dim_list,G_list,X0,y0,X1,y1,results_per_walker_t1):
    # Plot contour maps
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
        plt.title('$S = %.2f$' %(float(S)),fontsize=15)
        nfile='_landscape'+str(l)
        file1='contour_2d' + nfile + '.png'
        plt.savefig(file1,format='png',dpi=600)
        print('save 2d map plot to %s' %file1)
        cbar.remove()
        plt.close()
    if flag=='3d_landscape':
        print('Start: "plot(3d_landscape)"')
        # 3D plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        pnt3d_1=ax.plot_trisurf(dim_list[0],dim_list[1],G_list,linewidth=0,alpha=1.0,cmap='Greys')
        ax.set_axis_off()
        nfile='_landscape'+str(l)
        file1='3d_landscape' + nfile + '.png'
        plt.savefig(file1,format='png',dpi=600,bbox_inches='tight')
        print('save 3d map plot to %s' %file1)
        plt.close()
    # Plot contour map and points of t1 exploration
    elif flag=='t1_exploration':
        X1 = list(map(list, zip(*X1)))
        tim=np.arange(len(X1[0]))
        pnt3d_1=plt.tricontour(dim_list[0],dim_list[1],G_list,20,linewidths=1,colors='k')
        plt.clabel(pnt3d_1,inline=1,fontsize=5)
        pnt3d_2=plt.tricontourf(dim_list[0],dim_list[1],G_list,100,cmap='Greys')
        pnt3d_3=plt.scatter(X1[0][:],X1[1][:],c=tim,cmap='inferno',s=50,linewidth=1,zorder=4,alpha=0.8)
        pnt3d_4=plt.scatter(X0[0][:],X0[1][:],c='black',s=50,linewidth=1,zorder=4,alpha=0.8)
        cbar_2=plt.colorbar(pnt3d_3,pad=0.06)
        cbar_2.set_label("Time step \n", fontsize=12)
        cbar=plt.colorbar(pnt3d_2,pad=0.01)
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
        if w==adven_per_SPF-1:
            cbar.remove()
            cbar_2.remove()
        plt.close()
    # Plot boxplots with rmse for t1 analysis
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

##### END OTHER FUNCTIONS ######
################################################################################
# Measure initial time
start = time()
# Get initial values from input file
(dask_parallel, NCPU, verbosity_level, log_name,Nspf, 
        S, param, center_min, center_max, grid_min, grid_max, 
        grid_Delta, adven_per_SPF, adven, t1_time, d_threshold, 
        t0_time, ML, error_metric, CV, k_fold, test_last_proportion, 
        n_neighbor, weights, GBR_criterion, GBR_n_estimators, 
        GBR_learning_rate, GBR_max_depth, GBR_min_samples_split, 
        GBR_min_samples_leaf, GPR_alpha, GPR_length_scale, 
        GPR_alpha_lim, GPR_length_scale_lim, KRR_alpha, KRR_kernel,  
        KRR_gamma, optimize_KRR_hyperparams, optimize_GPR_hyperparams, 
        KRR_alpha_lim, KRR_gamma_lim, width_min, width_max, Amplitude_min, 
        Amplitude_max, N, t2_time, t2_ML, t2_exploration, t1_analysis, 
        diff_popsize, diff_tol, t2_train_time, calculate_grid, grid_name,
        plot_t1_exploration,plot_contour_map,plot_t1_error_metric,initial_spf) = read_initial_values(input_file_name)
# Run main program
main()
# Measure and print final time
time_taken = time()-start
print ('Process took %0.4f seconds' %time_taken)
