#!/usr/bin/env python3
# Marcos del Cueto
# Department of Chemistry and MIF, University of Liverpool
# Script to estimate smoothness 'S' from datasets
#################################################################################
import ast
import math
import numpy as np
import statistics
import matplotlib.pyplot as plt
#################################################################################
######   START CUSTOMIZABLE PARAMETERS ########
data_file_name = 'data_x_G'      # prefix of input file name
Nspf = 100                      # number of landscapes
adven = [80]           # array with adventurousness values
######   END CUSTOMIZABLE PARAMETERS   ######
#################################################################################
for a in adven:
    prediction_C = []
    F1 = []
    for l in range(Nspf):
        X  = []
        G  = []
        #F1 = []
        f_data = open('%s_%i_%s.dat' % (data_file_name,a,l), 'r')
        for line in f_data:
            provi_X = []
            line = ast.literal_eval(line)
            for i in range(len(line)-1):
                provi_X.append(line[i])
            provi_G = line[i+1]
            X.append(provi_X)
            G.append(provi_G)
        #print(X)
        #print(G)
        #print('#################')
        #for i in range(len(G)):
            #for j in range(len(G)):
        for i in range(20):
            for j in range(20):
                if j>i:
                    Delta_x = 0.0
                    Delta_G = np.sqrt((G[i] - G[j])**2)
                    for k in range(len(X[0])):
                        Delta_x = Delta_x + (X[i][k]-X[j][k])**2
                    Delta_x = np.sqrt(Delta_x)
                    if Delta_G > 0.0:
                        F1.append(Delta_G/Delta_x**1.0)
                        #F1.append(Delta_x/(2.0*Delta_G))

                    #print(i,j,Delta_G/Delta_x**1.0)

        av_f1 = statistics.mean(F1)
        prediction_C.append(av_f1)
        stdev_f1 = statistics.stdev(F1)
        #print('average F1:')
        #print(av_f1,'+/-', stdev_f1)
        #print(len(F1))
    final_pred_C_mean   = statistics.mean(prediction_C)
    final_pred_C_median = statistics.median(prediction_C)
    final_pred_C_stdev  = statistics.stdev(prediction_C)
    print('At a %i:' %(a))
    print('C:',final_pred_C_mean,final_pred_C_median,final_pred_C_stdev)
    final_pred_S_mean = 1/(2.0*final_pred_C_mean)
    final_pred_S_mean_error = final_pred_C_stdev/(2.0*final_pred_C_mean**2)
    final_pred_S_median = 1/(2.0*final_pred_C_median)
    final_pred_S_median_error = final_pred_C_stdev/(2.0*final_pred_C_median**2)
    print('S mean:', final_pred_S_mean,'+/-',final_pred_S_mean_error)
    print('S median:', final_pred_S_median,'+/-',final_pred_S_median_error)
