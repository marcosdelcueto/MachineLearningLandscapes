#!/usr/bin/env python3
# Marcos del Cueto
# Department of Chemistry and MIF, University of Liverpool
#################################################################################
#-21/01/2020
# Create MLL (Machine Learning Landscapes)
#################################################################################
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
######## START CUSTOMIZABLE PARAMETERS ########
### Parallel Computing ###
is_dask=True                                    # Decide if dask is used
### Verbose ###
print_logs=True                                 # Decide if verbose info is written into 'log_grid_l_XXX'
### Landscape parameters ###
Nspf=200					# Number of different landscapes generate
S=0.20                                          # Smoothness - minimum width of Gaussian functions
iseed=None                                      # Seed for random number to generate landscape. Default: None
param=2 					# Number of descriptors
center_min=0.00				 	# Minimum value for center of each Gaussian in each DOF
center_max=1.00				 	# Maximum value for center of each Gaussian in each DOF
### Grid parameters ###
grid_min=0.00                                   # Start point of grid in each DOF
grid_max=1.00					# End point of grid in each DOF
grid_Delta=0.01 				# Increment of grid in each DOF
### Exploration parameters ###
Nwalkers=10                    		        # Number of walkers per landscape
adven=[10,20,30,40,50,60,70,80,90,100]          # percentage of special points per walker
total_time=11                                   # Number of time steps for the walker
d_threshold=0.05               	                # Threshold maximum distance that at each time step
steps_unbiased=10                   	        # Number of initial unbiased steps
initial_sampling='different'                    # Initial points in a grid for different walkers
### ML algorithm ###
ML='kNN'                                        # Choose what ML algorithm is used
error_metric='rmse'                             # Choose which error metric to analyze
CV='loo'                                        # Choose cross-validation: 'kf' or 'loo' or 'sort'
k_fold=10                                       # If CV='kf': number of folds for k-fold cross validation
test_last_percentage=0.1                        # If CV='sort': percentage of last points chosen as test
if ML=='kNN':              		        ### k-Nearest Neighbors (k-NN)
    n_neighbor=3          		        # Number of nearest neighbors consider for kNN regression
    weights='distance'     		        # Type of weight used in kNN regression
elif ML=='GBR':            		        ### Gradient Boosting Regression (GBR)
    GBR_criterion='friedman_mse'                # function to measure quality of a split
    GBR_n_estimators=100                        # number of boosting stages to perform
    GBR_learning_rate=0.10                      # how much to shrink contribution of each tree
    GBR_max_depth=5                             # limits number of nodes in each tree
    GBR_min_samples_split=2                     # minimum number of samples required to split an internal node
    GBR_min_samples_leaf=2                      # minimum number of samples required to be at a leaf node
elif ML=='GPR':                                 ### Gaussian Process Regression (GPR). Kernel: RBF + noise
    A_RBF=1.0                                   # hyperparameter of RBF kernel
    A_noise=1.0                                 # hyperparameter of noise kernel
    GPR_alpha=0.001                             # parameter of GPR adding extra noise
    kernel_length_scale=1.0                     # hyperparameter length scale of RBF kernel
    kernel_noise_level=1.0                      # hyperparameter noise level of noise kernel
elif ML=='KRR':                                 ### Kernel Ridge Regression (GPR). Kernel: RBF
    KRR_alpha=1.0                               # regularization value: 0 would be equal to linear least square
    KRR_kernel='rbf'                            # type of kernel (for now, jusr 'rbf' available)
    KRR_gamma=1.0                               # hyperparameter controlling variance of kernel
    optimize_gamma=True                         # decide whether to optimize KRR_gamma
    KRR_gamma_lim = (0.01, 100.0)               # interval in which KRR_gamma is allowed to vary
### Possible initial values ###
allowed_initial_sampling=['same','different']   # possible values of 'initial_sampling'
allowed_CV=['kf','loo','sort']                  # possible types of cross-correlation
allowed_ML=[None,'kNN','GBR','GPR','KRR']       # possible ML algorithms
allowed_error_metric=['rmse']                   # possible error metrics to analyze
######## END CUSTOMIZABLE PARAMETERS ########
#################################################################################
width_min=S                                     # Minimum width of each Gaussian function
width_max=1.0/3.0                               # Maximum width of each Gaussian function
Amplitude_min=0.0                               # Minimum amplitude of each Gaussian function
Amplitude_max=1.0                               # Maximum amplitude of each Gaussian function
N=int(round((1/(S**param))))                    # Number of Gaussian functions of a specific landscape
if iseed==None: iseed=random.randrange(2**30-1) # If no seed is specified, choose a random one
#################################################################################
###### START MAIN ######
def main(iseed):
    check_input_values()
    results=[]
    if is_dask==True:
        for l in range(Nspf):
            iseed=iseed+l
            provi_result=delayed(MLL)(iseed,l,print_logs)
            results.append(provi_result)
        final_result=dask.compute(results,scheduler='processes')
        final_result_T=[list(i) for i in zip(*final_result[0])]
    elif is_dask==False:
        for l in range(Nspf):
            iseed=iseed+l
            provi_result=MLL(iseed,l,print_logs)
            results.append(provi_result)
        final_result=results
        final_result_T=[list(i) for i in zip(*final_result)]

    print('--- Final results:')
    for i in range(Nwalkers):
        print('-- Adventurousness: %6.1f' %(adven[i]))
        print('- RMSE:',final_result_T[i][:])
        print('- Mean: %f' %(statistics.mean(final_result_T[i])))

    if ML!=None: plot(error_metric,final_result_T)
###### END MAIN ######
#################################################################################

#################################################################################
###### START OTHER FUNCTIONS ######
def MLL(iseed,l,print_logs):
    error_metric_list=[]
    if print_logs==True:
        f_out = open('log_grid_l_%s' % l, 'w')
    else:
        f_out=None
    dim_list, G_list, Ngrid, max_G = generate_grid(iseed,l,f_out)
    for w in range(Nwalkers):
        X,y = explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G)
        if ML=='kNN': error_metric=kNN(X,y,iseed,l,w,f_out)
        if ML=='GBR': error_metric=GBR(X,y,iseed,l,w,f_out)
        if ML=='GPR': error_metric=GPR(X,y,iseed,l,w,f_out)
        if ML=='KRR':
            #hyperparams=[KRR_alpha,KRR_gamma]
            hyperparams=KRR_gamma
            if optimize_gamma == False:
                error_metric=KRR(hyperparams,X,y,iseed,l,w,f_out,print_logs)
            else:
                mini_args=(X,y,iseed,l,w,f_out,print_logs)
                #KRR_alpha_lim = (0.1, 100.0)
                #KRR_gamma_lim = (0.01, 100.0)
                #bounds = [KRR_alpha_lim] + [KRR_gamma_lim]
                bounds = [KRR_gamma_lim]
                solver=differential_evolution(KRR,bounds,args=mini_args,popsize=15,tol=0.01)
                best_hyperparams = solver.x
                best_rmse = solver.fun
                f_out.write("Best hyperparameters: %f \n" %best_hyperparams)
                f_out.write("Best rmse: %f \n"  %best_rmse)
                error_metric = best_rmse
        error_metric_list.append(error_metric)
    if print_logs==True: f_out.close()
    return error_metric_list

def generate_grid(iseed,l,f_out):
    Amplitude      = []
    center_N       = [[] for i in range(N)]
    width_N        = [[] for i in range(N)]
    x_list         = []
    dim_list       = [[] for i in range(param)]
    G_list         = []
    if print_logs==True: f_out.write('## Start: "generate_grid" function \n')
    if print_logs==True: f_out.write("########################### \n")
    if print_logs==True: f_out.write("##### Landscape', %i '##### \n" % (l))
    if print_logs==True: f_out.write("########################### \n")
    if print_logs==True: f_out.write("%s %i %s %6.2f \n" % ('Generated with seed:', iseed, ', and grid_max:', grid_max))
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
        if print_logs==True: f_out.write("%4s %14s %11s %13s \n" % ("N","Amplitude","Center","Width"))
        for i in range(len(Amplitude)):
            if print_logs==True: f_out.write("%4i %2s %10.6f %2s %10.6f %2s %10.6f \n" % (i, "", Amplitude[i],"",center_N[i][0],"",width_N[i][0]))
    elif param==2:
        if print_logs==True: f_out.write("%4s %14s %22s %35s \n" % ("N","Amplitude","Center","Width"))
        for i in range(len(Amplitude)):
            if print_logs==True: f_out.write("%4i %2s %10.6f %2s %10.6f %10.6f %2s %10.6f %10.6f \n" % (i, "", Amplitude[i],"",center_N[i][0],center_N[i][1],"",width_N[i][0],width_N[i][1]))
    elif param==3:
        if print_logs==True: f_out.write("%4s %14s %22s %34s \n" % ("N","Amplitude","Center","Width"))
        for i in range(len(Amplitude)):
            if print_logs==True: f_out.write("%4i %2s %10.6f %2s %10.6f %10.6f %10.6f %2s %10.6f %10.6f %10.6f \n" % (i, "", Amplitude[i],"",center_N[i][0],center_N[i][1],center_N[i][2],"",width_N[i][0],width_N[i][1],width_N[i][2]))
    elif param==4:
        if print_logs==True: f_out.write("%4s %14s %28s %45s \n" % ("N","Amplitude","Center","Width"))
        for i in range(len(Amplitude)):
            if print_logs==True: f_out.write("%4i %2s %10.6f %2s %10.6f %10.6f %10.6f %10.6f %2s %10.6f %10.6f %10.6f %10.6f \n" % (i, "", Amplitude[i],"",center_N[i][0],center_N[i][1],center_N[i][2],center_N[i][3],"",width_N[i][0],width_N[i][1],width_N[i][2],width_N[i][3]))
    elif param==5:
        if print_logs==True: f_out.write("%4s %14s %33s %57s \n" % ("N","Amplitude","Center","Width"))
        for i in range(len(Amplitude)):
            if print_logs==True: f_out.write("%4i %2s %10.6f %2s %10.6f %10.6f %10.6f %10.6f %10.6f %2s %10.6f %10.6f %10.6f %10.6f %10.6f \n" % (i, "", Amplitude[i],"",center_N[i][0],center_N[i][1],center_N[i][2],center_N[i][3],center_N[i][4],"",width_N[i][0],width_N[i][1],width_N[i][2],width_N[i][3],width_N[i][4]))

    # CALCULATE G GRID #
    counter=0
    for j in range(param):
        x_list.append(grid_min)
    if param==1:
        if print_logs==True: f_out.write("%8s %8s %11s \n" % ("i","x","G"))
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
            if print_logs==True: f_out.write("%8i %2s %6.2f %2s %10.6f \n" % (counter,"",dim_list[0][counter],"",G_list[counter]))
            counter=counter+1
            x_list[0]=x_list[0]+grid_Delta
    elif param==2:
        if print_logs==True: f_out.write("%8s %11s %15s \n" % ("i","x","G"))
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
                if print_logs==True: f_out.write("%8i %2s %6.2f %6.2f %2s %10.6f \n" % (counter,"",dim_list[0][counter],dim_list[1][counter],"",G_list[counter]))
                x_list[1]=x_list[1]+grid_Delta
                counter=counter+1
            x_list[0]=x_list[0]+grid_Delta
    elif param==3:
        if print_logs==True: f_out.write("%8s %14s %19s \n" % ("i","x","G"))
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
                    if print_logs==True: f_out.write("%8i %2s %6.2f %6.2f %6.2f %2s %10.6f \n" % (counter,"",dim_list[0][counter],dim_list[1][counter],dim_list[2][counter],"",G_list[counter]))
                    x_list[2]=x_list[2]+grid_Delta
                    counter=counter+1
                x_list[1]=x_list[1]+grid_Delta
            x_list[0]=x_list[0]+grid_Delta
    elif param==4:
        if print_logs==True: f_out.write("%8s %18s %22s \n" % ("i","x","G"))
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
                        if print_logs==True: f_out.write("%8i %2s %6.2f %6.2f %6.2f %6.2f %2s %10.6f \n" % (counter,"",dim_list[0][counter],dim_list[1][counter],dim_list[2][counter],dim_list[3][counter],"",G_list[counter]))
                        x_list[3]=x_list[3]+grid_Delta
                        counter=counter+1
                    x_list[2]=x_list[2]+grid_Delta
                x_list[1]=x_list[1]+grid_Delta
            x_list[0]=x_list[0]+grid_Delta
    elif param==5:
        if print_logs==True: f_out.write("%8s %22s %25s \n" % ("i","x","G"))
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
                            if print_logs==True: f_out.write("%8i %2s %6.2f %6.2f %6.2f %6.2f %6.2f %2s %10.6f \n" % (counter,"",dim_list[0][counter],dim_list[1][counter],dim_list[2][counter],dim_list[3][counter],dim_list[4][counter],"",G_list[counter]))
                            x_list[4]=x_list[4]+grid_Delta
                            counter=counter+1
                        x_list[3]=x_list[3]+grid_Delta
                    x_list[2]=x_list[2]+grid_Delta
                x_list[1]=x_list[1]+grid_Delta
            x_list[0]=x_list[0]+grid_Delta


    Ngrid=int((grid_max/grid_Delta+1)**param)   # calculate number of grid points
    max_G=max(G_list)
    min_G=min(G_list)
    if print_logs==True: f_out.write("Number of grid points: %i \n" %Ngrid)
    if print_logs==True: f_out.write("Maximum value of grid: %f \n" %max_G)
    if print_logs==True: f_out.write("Maximum value of grid: %f \n" %min_G)
    return dim_list, G_list, Ngrid, max_G

def check_input_values():
    if type(is_dask) != bool:
        print ('INPUT ERROR: is_dask should be boolean, but is:', is_dask)
        sys.exit()
    if type(print_logs) != bool:
        print ('INPUT ERROR: print_logs should be boolean, but is:', print_logs)
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
    print("\n")
    print('### START PRINT INPUT ###')
    print('# Parallel computing #')
    print('is_dask:',is_dask)
    print('# Verbose #')
    print('print_logs:',print_logs)
    print('# Landscape parameters #')
    print('Nspf:',Nspf)
    print('S:',S)
    print('iseed:',iseed)
    print('param:',param)
    print('center_min:',center_min)
    print('center_max:',center_max)
    print('Grid parameters')
    print('grid_min:',grid_min)
    print('grid_max:',grid_max)
    print('grid_Delta:',grid_Delta)
    print('# Exploration parameters #')
    print('Nwalkers:',Nwalkers)
    print('adven:',adven)
    print('total_time:',total_time)
    print('d_threshold:',d_threshold)
    print('steps_unbiased:',steps_unbiased)
    print('initial_sampling:',initial_sampling)
    print('allowed_initial_sampling:',allowed_initial_sampling)
    print('# ML algorithm #')
    print('allowed_ML:',allowed_ML)
    print('allowed_error_metric:',allowed_error_metric)
    print('allowed_CV:',allowed_CV)
    print('ML:',ML)
    print('error_metric:',error_metric)
    print('CV:',CV)
    print('k_fold:',k_fold)
    print('test_last_percentage:',test_last_percentage)
    if ML=='kNN':
        print('n_neighbor:',n_neighbor)
        print('weights:',weights)
    if ML=='GBR':
        print('GBR_criterion:',GBR_criterion)
        print('GBR_n_estimators:',GBR_n_estimators)
        print('GBR_learning_rate:',GBR_learning_rate)
        print('GBR_max_depth:',GBR_max_depth)
        print('GBR_min_samples_split:',GBR_min_samples_split)
        print('GBR_min_samples_leaf:',GBR_min_samples_leaf)
    if ML=='GPR':
        print('A_RBF:',A_RBF)
        print('A_noise:',A_noise)
        print('GPR_alpha:',GPR_alpha)
        print('kernel_length_scale:',kernel_length_scale)
        print('kernel_noise_level:',kernel_noise_level)
    print('#### END PRINT INPUT ####')
    print("\n")

def explore_landscape(iseed,l,w,dim_list,G_list,f_out,Ngrid,max_G):
    walker_x       = []
    path_x         = [[] for i in range(param)]
    path_G         = []
    list_t         = []
    prob           = []
    neighbor_walker= [[] for i in range(param)]
    neighbor_G     = []
    # START UNBIASED SWARM #
    if print_logs==True: f_out.write('## Start: "explore_landscape" function \n')
    if print_logs==True: f_out.write("############# \n")
    if print_logs==True: f_out.write("Start swarm %i \n" % (w))
    if print_logs==True: f_out.write("Adventurousness: %f \n" % (adven[w]))
    if print_logs==True: f_out.write("############# \n")
    Nx=((grid_max-grid_min)/grid_Delta)+1
    if print_logs==True: f_out.write("Number of points per dimension: %i \n" %Nx)
    if print_logs==True: f_out.write("Testing w: %i, iseed: %i \n" % (w,iseed))
    for t in range(steps_unbiased):
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
        if param==1 and print_logs==True: f_out.write("timestep %4i %2s %6.2f %2s %10.6f \n" % (t,"",walker_x[0],"",G_list[num_in_grid]))
        if param==2 and print_logs==True: f_out.write("timestep %4i %2s %6.2f %6.2f %2s %10.6f \n" % (t,"",walker_x[0],walker_x[1],"",G_list[num_in_grid]))
        if param==3 and print_logs==True: f_out.write("timestep %4i %2s %6.2f %6.2f %6.2f %2s %10.6f \n" % (t,"",walker_x[0],walker_x[1],walker_x[2],"",G_list[num_in_grid]))
        if param==4 and print_logs==True: f_out.write("timestep %4i %2s %6.2f %6.2f %6.2f %6.2f %2s %10.6f \n" % (t,"",walker_x[0],walker_x[1],walker_x[2],walker_x[3],"",G_list[num_in_grid]))
        if param==5 and print_logs==True: f_out.write("timestep %4i %2s %6.2f %6.2f %6.2f %6.2f %6.2f %2s %10.6f\n" % (t,"",walker_x[0],walker_x[1],walker_x[2],walker_x[3],walker_x[4],"",G_list[num_in_grid]))
    # CONTINUE BIASED SWARM #
    nfile='landscape'+str(l)+'_swarm'+str(w)
    x_param=[[] for j in range(param)]
    y=[]
    for i in range(param):
        walker_x[i]=path_x[i][steps_unbiased-1]
        del path_x[i][steps_unbiased:]
    del path_G[steps_unbiased:]
    del list_t[steps_unbiased:]
    for t in range(steps_unbiased,total_time):
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
        if print_logs==True: f_out.write("Special points: %i \n" % (special_points))
        for i in range(special_points):
            if param==1 and print_logs==True: f_out.write("%i %6.2f %10.6f \n" % (i,minimum_path_x[0][i],minimum_path_G[i]))
            if param==2 and print_logs==True: f_out.write("%i %6.2f %6.2f %10.6f \n" % (i,minimum_path_x[0][i],minimum_path_x[1][i],minimum_path_G[i]))
            if param==3 and print_logs==True: f_out.write("%i %6.2f %6.2f %6.2f %10.6f \n" % (i,minimum_path_x[0][i],minimum_path_x[1][i],minimum_path_x[2][i],minimum_path_G[i]))
            if param==4 and print_logs==True: f_out.write("%i %6.2f %6.2f %6.2f %6.2f %10.6f \n" % (i,minimum_path_x[0][i],minimum_path_x[1][i],minimum_path_x[2][i],minimum_path_x[3][i],minimum_path_G[i]))
            if param==5 and print_logs==True: f_out.write("%i %6.2f %6.2f %6.2f %6.2f %6.2f %10.6f \n" % (i,minimum_path_x[0][i],minimum_path_x[1][i],minimum_path_x[2][i],minimum_path_x[3][i],minimum_path_x[4][i],minimum_path_G[i]))
            if param==1 and print_logs==True: f_out.write("Selected point draw: %f %f \n" % (minimum_path_x[0][draw],minimum_path_G[draw]))
            if param==2 and print_logs==True: f_out.write("Selected point draw: %f %f %f \n" % (minimum_path_x[0][draw],minimum_path_x[1][draw],minimum_path_G[draw]))
            if param==3 and print_logs==True: f_out.write("Selected point draw: %f %f %f %f \n" % (minimum_path_x[0][draw],minimum_path_x[1][draw],minimum_path_x[2][draw],minimum_path_G[draw]))
            if param==4 and print_logs==True: f_out.write("Selected point draw: %f %f %f %f %f \n" % (minimum_path_x[0][draw],minimum_path_x[1][draw],minimum_path_x[2][draw],minimum_path_x[3][draw],minimum_path_G[draw]))
            if param==5 and print_logs==True: f_out.write("Selected point draw: %f %f %f %f %f %f \n" % (minimum_path_x[0][draw],minimum_path_x[1][draw],minimum_path_x[2][draw],minimum_path_x[3][draw],minimum_path_x[4][draw],minimum_path_G[draw]))
        for i in range(param):
            if minimum_path_x[i][draw] != dim_list[i][draw_in_grid]:
                if print_logs==True: f_out.write("STOP - ERROR: minimum_path not equal to dum_list (maybe more than 1 point with that value in grid) \n")
                if param==1 and print_logs==True: f_out.write("Selected point draw in grid: %f %f \n" % (dim_list[0][draw_in_grid],G_list[draw_in_grid]))
                if param==2 and print_logs==True: f_out.write("Selected point draw in grid: %f %f %f \n" % (dim_list[0][draw_in_grid],dim_list[1][draw_in_grid],G_list[draw_in_grid]))
                if param==3 and print_logs==True: f_out.write("Selected point draw in grid: %f %f %f %f \n" % (dim_list[0][draw_in_grid],dim_list[1][draw_in_grid],dim_list[2][draw_in_grid],G_list[draw_in_grid]))
                if param==4 and print_logs==True: f_out.write("Selected point draw in grid: %f %f %f %f %f \n" % (dim_list[0][draw_in_grid],dim_list[1][draw_in_grid],dim_list[2][draw_in_grid],dim_list[3][draw_in_grid],G_list[draw_in_grid]))
                if param==5 and print_logs==True: f_out.write("Selected point draw in grid: %f %f %f %f %f %f \n" % (dim_list[0][draw_in_grid],dim_list[1][draw_in_grid],dim_list[2][draw_in_grid],dim_list[3][draw_in_grid],dim_list[4][draw_in_grid],G_list[draw_in_grid]))
                sys.exit()
        P=int(round(d_threshold/grid_Delta + 1))
        if print_logs==True: f_out.write("Consider nearby points: \n")
        counter3=0
        for i in range((P*2+1)**param):
                prob.append(0.0)
                neighbor_G.append(0.0)
                for j in range(param):
                    neighbor_walker[j].append(0.0)
        if param==1:
            if print_logs==True: f_out.write("%6s %5s %11s %11s %13s \n" % ("i","x","G","Prob","distance"))
            for i1 in range(-P,P+1):
                try:
                    index0=int(round(draw_in_grid-i1))
                    indexG=int(round(draw_in_grid-i1))
                    d_ij=round((math.sqrt((minimum_path_x[0][draw]-dim_list[0][index0])**2)),6)
                    if d_ij < d_threshold and d_ij > 0.0:
                        neighbor_walker[0][counter3]=dim_list[0][index0]
                        neighbor_G[counter3]=G_list[indexG]
                        prob[counter3]=1.0
                        prob_sum=prob_sum+prob[counter3]
                    if print_logs==True: f_out.write("%6i %6.2f %2s %10.6f %2s %5.1f %2s %10.6f \n" % (counter3,dim_list[0][index0],"",G_list[indexG],"",prob[counter3],"",d_ij))
                except:
                    pass
                counter3=counter3+1
        elif param==2:
            if print_logs==True: f_out.write("%6s %8s %15s %11s %13s \n" % ("i","x","G","Prob","distance"))
            for i2 in range(-P,P+1):
                for i1 in range(-P,P+1):
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
                        if print_logs==True: f_out.write("%6i %6.2f %6.2f %2s %10.6f %2s %5.1f %2s %10.6f \n" % (counter3,dim_list[0][index0],dim_list[1][index1],"",G_list[indexG],"",prob[counter3],"",d_ij))
                    except:
                        pass
                    counter3=counter3+1
        elif param==3:
            if print_logs==True: f_out.write("%6s %11s %19s %12s %12s \n" % ("i","x","G","Prob","distance"))
            for i3 in range(-P,P+1):
                for i2 in range(-P,P+1):
                    for i1 in range(-P,P+1):
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
                            if print_logs==True: f_out.write("%6i %6.2f %6.2f %6.2f %2s %10.6f %2s %5.1f %2s %10.6f \n" % (counter3,dim_list[0][index0],dim_list[1][index1],dim_list[2][index2],"",G_list[indexG],"",prob[counter3],"",d_ij))
                        except:
                            pass
                        counter3=counter3+1
        elif param==4:
            if print_logs==True: f_out.write("%6s %15s %22s %12s %12s \n" % ("i","x","G","Prob","distance"))
            for i4 in range(-P,P+1):
                for i3 in range(-P,P+1):
                    for i2 in range(-P,P+1):
                        for i1 in range(-P,P+1):
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
                                if print_logs==True: f_out.write("%6i %6.2f %6.2f %6.2f %6.2f %2s %10.6f %2s %5.1f %2s %10.6f \n" % (counter3,dim_list[0][index0],dim_list[1][index1],dim_list[2][index2],dim_list[3][index3],"",G_list[indexG],"",prob[counter3],"",d_ij))
                            except:
                                pass
                            counter3=counter3+1
        elif param==5:
            if print_logs==True: f_out.write("%6s %19s %25s %12s %12s \n" % ("i","x","G","Prob","distance"))
            for i5 in range(-P,P+1):
                for i4 in range(-P,P+1):
                    for i3 in range(-P,P+1):
                        for i2 in range(-P,P+1):
                            for i1 in range(-P,P+1):
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
                                    if print_logs==True: f_out.write("%6i %6.2f %6.2f %6.2f %6.2f %6.2f %2s %10.6f %2s %5.1f %2s %10.6f \n" % (counter3,dim_list[0][index0],dim_list[1][index1],dim_list[2][index2],dim_list[3][index3],dim_list[4][index4],"",G_list[indexG],"",prob[counter3],"",d_ij))
                                except:
                                    pass
                                counter3=counter3+1
        if len(range((P*2+1)**param)) != len(prob):
            print("STOP - ERROR: Problem with number of nearby points considered for next step")
            sys.exit()
        if print_logs==True: f_out.write("Number of points considered: %i \n" % (len(range((P*2+1)**param))))
        if print_logs==True: f_out.write("Points within threshold: %f \n" % int(round((prob_sum))))

        for i in range(counter3):
            prob[i]=prob[i]/prob_sum
        draw=int(choice(range((P*2+1)**param),size=1,p=prob))
        for i in range(param):
            walker_x[i]=neighbor_walker[i][draw]
            path_x[i].append(walker_x[i])
        path_G.append(neighbor_G[draw])
        list_t.append(t)
        if print_logs==True: f_out.write("We draw neighbor no.: %6i\n" % (draw))
        if param==1 and print_logs==True: f_out.write("timestep %6i %2s %6.2f %2s %10.6f\n" % (t,"",walker_x[0],"",neighbor_G[draw]))
        if param==2 and print_logs==True: f_out.write("timestep %6i %2s %6.2f %6.2f %2s %10.6f\n" % (t,"",walker_x[0],walker_x[1],"",neighbor_G[draw]))
        if param==3 and print_logs==True: f_out.write("timestep %6i %2s %6.2f %6.2f %6.2f %2s %10.6f\n" % (t,"",walker_x[0],walker_x[1],walker_x[2],"",neighbor_G[draw]))
        if param==4 and print_logs==True: f_out.write("timestep %6i %2s %6.2f %6.2f %6.2f %6.2f %2s %10.6f\n" % (t,"",walker_x[0],walker_x[1],walker_x[2],walker_x[3],"",neighbor_G[draw]))
        if param==5 and print_logs==True: f_out.write("timestep %6i %2s %6.2f %6.2f %6.2f %6.2f %6.2f %2s %10.6f\n" % (t,"",walker_x[0],walker_x[1],walker_x[2],walker_x[3],walker_x[4],"",neighbor_G[draw]))
        for i in range(param):
            x_param[i].append(walker_x[i])
        y.append(neighbor_G[draw])
    if param==1:
        zippedList = list(zip(x_param[0],y))
        df=pd.DataFrame(zippedList,columns=['x_param[0]','y'])
        #df2=df.drop_duplicates(subset='y', keep="last")
        df2=df
        df_X=df2[['x_param[0]']]
        df_y=df2['y']
    elif param==2:
        zippedList = list(zip(x_param[0],x_param[1],y))
        df=pd.DataFrame(zippedList,columns=['x_param[0]','x_param[1]','y'])
        #df2=df.drop_duplicates(subset='y', keep="last")
        df2=df
        df_X=df2[['x_param[0]','x_param[1]']]
        df_y=df2['y']
    elif param==3:
        zippedList = list(zip(x_param[0],x_param[1],x_param[2],y))
        df=pd.DataFrame(zippedList,columns=['x_param[0]','x_param[1]','x_param[2]','y'])
        #df2=df.drop_duplicates(subset='y', keep="last")
        df2=df
        df_X=df2[['x_param[0]','x_param[1]','x_param[2]']]
        df_y=df2['y']
    elif param==4:
        zippedList = list(zip(x_param[0],x_param[1],x_param[2],x_param[3],y))
        df=pd.DataFrame(zippedList,columns=['x_param[0]','x_param[1]','x_param[2]','x_param[3]','y'])
        #df2=df.drop_duplicates(subset='y', keep="last")
        df2=df
        df_X=df2[['x_param[0]','x_param[1]','x_param[2]','x_param[3]']]
        df_y=df2['y']
    elif param==5:
        zippedList = list(zip(x_param[0],x_param[1],x_param[2],x_param[3],x_param[4],y))
        df=pd.DataFrame(zippedList,columns=['x_param[0]','x_param[1]','x_param[2]','x_param[3]','x_param[4]','y'])
        #df2=df.drop_duplicates(subset='y', keep="last")
        df2=df
        df_X=df2[['x_param[0]','x_param[1]','x_param[2]','x_param[3]','x_param[3]']]
        df_y=df2['y']
    X=df_X.to_numpy()
    y=df_y.to_numpy()
    if print_logs==True: f_out.write("## X: \n")
    if print_logs==True: f_out.write("%s \n" % (str(X)))
    if print_logs==True: f_out.write("## y: \n")
    if print_logs==True: f_out.write("%s \n" % (str(y)))
    return X,y


# CALCULATE k-NN #
def kNN(X,y,iseed,l,w,f_out):
    iseed=iseed+1
    if print_logs==True: f_out.write('## Start: "kNN" function \n')
    if print_logs==True: f_out.write('-------- \n')
    if print_logs==True: f_out.write('Perform k-NN \n')
    if print_logs==True: f_out.write('k= %i \n' % (n_neighbor))
    if print_logs==True: f_out.write('cross_validation %i - fold \n' % (k_fold))
    if print_logs==True: f_out.write('weights %s \n' % (weights))
    if print_logs==True: f_out.write('iseed %s \n' % (iseed))
    if print_logs==True: f_out.write('-------- \n')

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
                if print_logs==True: f_out.write('Landscape %i . Adventurousness: %i . k-fold: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],counter,r_pearson,rmse))
                if print_logs==True: f_out.write("%i test points: %s \n" % (len(test_index),str(test_index)))
                counter=counter+1
                average_r_pearson=average_r_pearson+r_pearson
                average_rmse=average_rmse+rmse
        if CV=='kf':
            average_r_pearson=average_r_pearson/k_fold
            average_rmse=average_rmse/k_fold
            if print_logs==True: f_out.write('k-fold average r_pearson score: %f \n' % (average_r_pearson))
            if print_logs==True: f_out.write('k-fold average rmse score: %f \n' % (average_rmse))
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
        if  print_logs==True: f_out.write("Train with first %i points \n" % (len(X_train)))
        if  print_logs==True: f_out.write("%s \n" % (str(X_train)))
        if  print_logs==True: f_out.write("Test with last %i points \n" % (len(X_test)))
        if  print_logs==True: f_out.write("%s \n" % (str(X_test)))
        if print_logs==True: f_out.write('Landscape %i . Adventurousness: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],total_r_pearson,total_rmse))
    if print_logs==True: f_out.write('Final r_pearson score: %f \n' % (total_r_pearson))
    if print_logs==True: f_out.write('Final rmse score: %f \n' % (total_rmse))
    if error_metric=='rmse': result=total_rmse
    return result

# CALCULATE GBR #
def GBR(X,y,iseed,l,w,f_out):
    iseed=iseed+1
    if print_logs==True: f_out.write('## Start: "GBR" function \n')
    if print_logs==True: f_out.write('-------- \n')
    if print_logs==True: f_out.write('Perform GBR\n')
    if print_logs==True: f_out.write('cross_validation %i - fold\n' % (k_fold))
    if print_logs==True: f_out.write('GBR criterion: %s\n' % (GBR_criterion))
    if print_logs==True: f_out.write('Number of estimators: %i\n' % (GBR_n_estimators))
    if print_logs==True: f_out.write('Learning rate: %f\n' % (GBR_learning_rate))
    if print_logs==True: f_out.write('Tree max depth: %i\n' % (GBR_max_depth))
    if print_logs==True: f_out.write('Min samples to split: %i\n' % (GBR_min_samples_split))
    if print_logs==True: f_out.write('Min samples per leaf: %i\n' % (GBR_min_samples_leaf))
    if print_logs==True: f_out.write('--------\n')

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
                if print_logs==True: f_out.write('Landscape %i . Adventurousness: %i . k-fold: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],counter,r_pearson,rmse))
                if print_logs==True: f_out.write("%i test points: %s \n" % (len(test_index),str(test_index)))
                counter=counter+1
                average_r_pearson=average_r_pearson+r_pearson
                average_rmse=average_rmse+rmse
        if CV=='kf':
            average_r_pearson=average_r_pearson/k_fold
            average_rmse=average_rmse/k_fold
            if print_logs==True: f_out.write('k-fold average r_pearson score: %f \n' % (average_r_pearson))
            if print_logs==True: f_out.write('k-fold average rmse score: %f \n' % (average_rmse))
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
        if  print_logs==True: f_out.write("Train with first %i points \n" % (len(X_train)))
        if  print_logs==True: f_out.write("%s \n" % (str(X_train)))
        if  print_logs==True: f_out.write("Test with last %i points \n" % (len(X_test)))
        if  print_logs==True: f_out.write("%s \n" % (str(X_test)))
        if print_logs==True: f_out.write('Landscape %i . Adventurousness: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],total_r_pearson,total_rmse))
    if print_logs==True: f_out.write('Final r_pearson score: %f \n' % (total_r_pearson))
    if print_logs==True: f_out.write('Final rmse score: %f \n' % (total_rmse))
    if error_metric=='rmse': result=total_rmse
    return result


# CALCULATE GPR #
def GPR(X,y,iseed,l,w,f_out):
    iseed=iseed+1
    if print_logs==True: f_out.write('## Start: "GPR" function \n')
    if print_logs==True: f_out.write('-------- \n')
    if print_logs==True: f_out.write('Perform GPR\n')
    if print_logs==True: f_out.write('Cross_validation %i - fold\n' % (k_fold))
    if print_logs==True: f_out.write('Initial A_RBF %f: \n' % (A_RBF))
    if print_logs==True: f_out.write('Initial kernel_length_scale: %f: \n' % (kernel_length_scale))
    if print_logs==True: f_out.write('Initial kernel_noise_level: %f: \n' % (kernel_noise_level))
    if print_logs==True: f_out.write('-------- \n')
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
            if print_logs==True: f_out.write('TEST X_train: \n')
            if print_logs==True: f_out.write('%s \n' % (str(X_train)))
            if print_logs==True: f_out.write('TEST y_train: \n')
            if print_logs==True: f_out.write('%s \n' % (str(y_train)))
            if print_logs==True: f_out.write('TEST X_test: \n')
            if print_logs==True: f_out.write('%s \n' % (str(X_test)))
            if print_logs==True: f_out.write('TEST y_test: \n')
            if print_logs==True: f_out.write('%s \n' % (str(y_test)))
            if print_logs==True: f_out.write('Converged kernel hyperparameters: %s \n' % (str(GPR.kernel_)))
            if print_logs==True: f_out.write('Converged alpha: %s \n' % (str(GPR.alpha_)))
            if print_logs==True: f_out.write('Parameters GPR: \n')
            if print_logs==True: f_out.write('%s \n' % (str(GPR.get_params(deep=True))))
            if print_logs==True: f_out.write('Parameters GPR kernel: \n')
            if print_logs==True: f_out.write('%s \n' % (str(GPR.kernel_.get_params(deep=True))))
            if print_logs==True: f_out.write('GPR X_train: \n')
            if print_logs==True: f_out.write('%s \n' % (str(GPR.X_train_)))
            if print_logs==True: f_out.write('GPR y_train: \n')
            if print_logs==True: f_out.write('%s \n' % (str(GPR.y_train_)))
            if print_logs==True: f_out.write('TEST X_train_scaled: \n')
            if print_logs==True: f_out.write('%s \n' % (str(X_train_scaled)))
            if print_logs==True: f_out.write('TEST X_test_scaled: \n')
            if print_logs==True: f_out.write('%s \n' % (str(X_test_scaled)))
            if print_logs==True: f_out.write('TEST y_test: \n')
            if print_logs==True: f_out.write('%s \n' % (str(y_test)))
            if print_logs==True: f_out.write('TEST y_pred: \n')
            if print_logs==True: f_out.write('%s \n' % (str(y_pred)))

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
                if print_logs==True: f_out.write('Landscape %i . Adventurousness: %i . k-fold: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],counter,r_pearson,rmse))
                if print_logs==True: f_out.write("%i test points: %s \n" % (len(test_index),str(test_index)))
                counter=counter+1
                average_r_pearson=average_r_pearson+r_pearson
                average_rmse=average_rmse+rmse
            if CV=='kf':
                r_pearson,_=pearsonr(y_test,y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                if print_logs==True: f_out.write('Landscape %i . Adventurousness: %i . k-fold: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],counter,r_pearson,rmse))
                if print_logs==True: f_out.write("%i test points: %s \n" % (len(test_index),str(test_index)))
                counter=counter+1
                average_r_pearson=average_r_pearson+r_pearson
                average_rmse=average_rmse+rmse
        if CV=='kf':
            average_r_pearson=average_r_pearson/k_fold
            average_rmse=average_rmse/k_fold
            if print_logs==True: f_out.write('k-fold average r_pearson score: %f \n' % (average_r_pearson))
            if print_logs==True: f_out.write('k-fold average rmse score: %f \n' % (average_rmse))
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
        if print_logs==True: f_out.write('TEST X_train: \n')
        if print_logs==True: f_out.write('%s \n' % (str(X_train)))
        if print_logs==True: f_out.write('TEST y_train: \n')
        if print_logs==True: f_out.write('%s \n' % (str(y_train)))
        if print_logs==True: f_out.write('TEST X_test: \n')
        if print_logs==True: f_out.write('%s \n' % (str(X_test)))
        if print_logs==True: f_out.write('TEST y_test: \n')
        if print_logs==True: f_out.write('%s \n' % (str(y_test)))
        if print_logs==True: f_out.write('Converged kernel hyperparameters: %s \n' % (str(GPR.kernel_)))
        if print_logs==True: f_out.write('Converged alpha: %s \n' % (str(GPR.alpha_)))
        if print_logs==True: f_out.write('Parameters GPR: \n')
        if print_logs==True: f_out.write('%s \n' % (str(GPR.get_params(deep=True))))
        if print_logs==True: f_out.write('Parameters GPR kernel: \n')
        if print_logs==True: f_out.write('%s \n' % (str(GPR.kernel_.get_params(deep=True))))
        if print_logs==True: f_out.write('GPR X_train: \n')
        if print_logs==True: f_out.write('%s \n' % (str(GPR.X_train_)))
        if print_logs==True: f_out.write('GPR y_train: \n')
        if print_logs==True: f_out.write('%s \n' % (str(GPR.y_train_)))
        if print_logs==True: f_out.write('TEST X_train_scaled: \n')
        if print_logs==True: f_out.write('%s \n' % (str(X_train_scaled)))
        if print_logs==True: f_out.write('TEST X_test_scaled: \n')
        if print_logs==True: f_out.write('%s \n' % (str(X_test_scaled)))
        if print_logs==True: f_out.write('TEST y_test: \n')
        if print_logs==True: f_out.write('%s \n' % (str(y_test)))
        if print_logs==True: f_out.write('TEST y_pred: \n')
        if print_logs==True: f_out.write('%s \n' % (str(y_pred)))

        total_r_pearson,_=pearsonr(y_test,y_pred)
        mse = mean_squared_error(y_test, y_pred)
        total_rmse = np.sqrt(mse)
        if  print_logs==True: f_out.write("Train with first %i points \n" % (len(X_train)))
        if  print_logs==True: f_out.write("%s \n" % (str(X_train)))
        if  print_logs==True: f_out.write("Test with last %i points \n" % (len(X_test)))
        if  print_logs==True: f_out.write("%s \n" % (str(X_test)))
        if print_logs==True: f_out.write('Landscape %i . Adventurousness: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],total_r_pearson,total_rmse))
    if print_logs==True: f_out.write('Final r_pearson score: %f \n' % (total_r_pearson))
    if print_logs==True: f_out.write('Final rmse score: %f \n' % (total_rmse))
    if error_metric=='rmse': result=total_rmse
    return result

# CALCULATE KRR #
def KRR(hyperparams,X,y,iseed,l,w,f_out,print_logs):
    #KRR_alpha, KRR_gamma = hyperparams
    KRR_gamma = hyperparams
    iseed=iseed+1
    if print_logs==True: f_out.write('## Start: "KRR" function \n')
    if print_logs==True: f_out.write('-------- \n')
    if print_logs==True: f_out.write('Perform KRR\n')
    if print_logs==True: f_out.write('Cross_validation %i - fold\n' % (k_fold))
    if print_logs==True: f_out.write('KRR alpha %f\n' % (KRR_alpha))
    if print_logs==True: f_out.write('KRR gamma %f\n' % (KRR_gamma))
    if print_logs==True: f_out.write('KRR kernel %s\n' % (KRR_kernel))
    if print_logs==True: f_out.write('-------- \n')
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
            #if print_logs==True: f_out.write('Parameters KRR: \n')
            #if print_logs==True: f_out.write('%s \n' % (str(KRR.get_params(deep=True))))

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
                if print_logs==True: f_out.write('Landscape %i . Adventurousness: %i . k-fold: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],counter,r_pearson,rmse))
                if print_logs==True: f_out.write("%i test points: %s \n" % (len(test_index),str(test_index)))
                counter=counter+1
                average_r_pearson=average_r_pearson+r_pearson
                average_rmse=average_rmse+rmse
        if CV=='kf':
            average_r_pearson=average_r_pearson/k_fold
            average_rmse=average_rmse/k_fold
            if print_logs==True: f_out.write('k-fold average r_pearson score: %f \n' % (average_r_pearson))
            if print_logs==True: f_out.write('k-fold average rmse score: %f \n' % (average_rmse))
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
        if  print_logs==True: f_out.write("Train with first %i points \n" % (len(X_train)))
        if  print_logs==True: f_out.write("%s \n" % (str(X_train)))
        if  print_logs==True: f_out.write("Test with last %i points \n" % (len(X_test)))
        if  print_logs==True: f_out.write("%s \n" % (str(X_test)))
        if print_logs==True: f_out.write('Landscape %i . Adventurousness: %i . r_pearson: %f . rmse: %f \n' % (l,adven[w],total_r_pearson,total_rmse))
    if print_logs==True: f_out.write('Parameters KRR: \n')
    if print_logs==True: f_out.write('%s \n' % (str(KRR.get_params(deep=True))))
    if print_logs==True: f_out.write('Final r_pearson score: %f \n' % (total_r_pearson))
    if print_logs==True: f_out.write('Final rmse score: %f \n' % (total_rmse))
    if error_metric=='rmse': result=total_rmse
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
        print('save rmse box plot to %s' %file1)
        plt.close()

        pntbox=plt.boxplot(final_result_T,patch_artist=True,labels=adven,showfliers=True)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel('Adventurousness (%)',fontsize=15)
        plt.ylabel('RMSE (arb. units)',fontsize=15)
        file1='rmse_with_outliers.png'
        plt.savefig(file1,format='png',dpi=600)
        print('save rmse box plot to %s' %file1)
        plt.close()



##### END OTHER FUNCTIONS ######
################################################################################
start = time()
main(iseed)
time_taken = time()-start
print ('Process took %0.2f seconds' %time_taken)
