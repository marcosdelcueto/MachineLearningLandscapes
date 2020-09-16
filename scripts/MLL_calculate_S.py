#!/usr/bin/env python3
# Marcos del Cueto
# Department of Chemistry and MIF, University of Liverpool
#################################################################################
# Script to estimate smoothness 'S' from datasets
#################################################################################
import ast
import math
import numpy as np
import statistics
import matplotlib.pyplot as plt
#################################################################################
######   START CUSTOMIZABLE PARAMETERS ######
data_file_name = 'data_x_G'                 # prefix of input file name
Nspf = 100                                  # number of landscapes
adven = [10,40,70,100]                      # array with adventurousness values
scaling_factor = 0.45                       # scaling factor to transform G corrugation to our arbitrary S scale
######   END CUSTOMIZABLE PARAMETERS   ######
#################################################################################
# Loop for each a values
for a in adven:
    prediction_C = []
    # Loop for each dataset
    for l in range(Nspf):
        corrugation = []
        X  = []
        G  = []
        f_data = open('%s_%i_%s.dat' % (data_file_name,a,l), 'r')
        # Read data
        for line in f_data:
            provi_X = []
            line = ast.literal_eval(line)
            for i in range(len(line)-1):
                provi_X.append(line[i])
            provi_G = line[i+1]
            X.append(provi_X)
            G.append(provi_G)
        # Calculate list with all corrugation values, as Delta_G/Delta_x, for each dataset
        for i in range(len(G)):
            for j in range(len(G)):
                if j>i:
                    Delta_x = 0.0
                    Delta_G = np.sqrt((G[i] - G[j])**2)
                    for k in range(len(X[0])):
                        Delta_x = Delta_x + (X[i][k]-X[j][k])**2
                    Delta_x = np.sqrt(Delta_x)
                    if Delta_G > 0.0:
                        corrugation.append(Delta_G/Delta_x)
        # Calculate average corrigation for each dataset
        av_corrugation = statistics.mean(corrugation)
        prediction_C.append(av_corrugation)
    # Calculate final averaged corrugation values
    final_pred_C_median = statistics.median(prediction_C)
    final_pred_C_stdev  = statistics.stdev(prediction_C)
    # Calculate Smoothness (S) as inverse of corrugation, times a scaling factor
    final_pred_S_median = 1/(final_pred_C_median) * scaling_factor
    final_pred_S_median_error = final_pred_C_stdev/(final_pred_C_median**2) * scaling_factor
    print('At a = %4i - S Median: %7.4f. S Stdev: %7.4f' % (a,final_pred_S_median,final_pred_S_median_error))
