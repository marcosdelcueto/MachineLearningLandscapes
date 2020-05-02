#!/usr/bin/env python3
# Marcos del Cueto
# Department of Chemistry and MIF, University of Liverpool
#################################################################################
import ast
import numpy as np
import statistics
#################################################################################
######   START CUSTOMIZABLE PARAMETERS ########
data_file_name = 'data_x_G'      # prefix of input file name
Nspf = 100                       # number of landscapes
adven = [10,20,30,40,50,60,70,80,90,100]     # array with adventurousness values
######   END CUSTOMIZABLE PARAMETERS   ######
#################################################################################
final_S = []
error_S = []
final_a = []
error_a = []

# Set value of general S factors
factor_F1 = 0.27
S_C1 =  0.78
S_C2 = -0.50
S_C3 =  1.00
# Set value of general a factors
factor_F2 = 1.0
a_C1 =  1.0
a_C2 =  0.001
a_C3 =  1.00

# Loop for each adventurousness
for a in adven:
    # initialize final arrays
    average_F1 = []
    average_F2 = []
    error_F1 = []
    error_F2 = []
    # loop to average over all landscapes of a given adventurousness
    for l in range(Nspf):
        # Initialize arrays
        X  = []
        G  = []
        d0 = []
        F1 = []
        F2 = []
        # read data and assign it to X and G arrays
        f_data = open('%s_%i_%s.dat' % (data_file_name,a,l), 'r')
        for line in f_data:
            provi_X = []
            line = ast.literal_eval(line)
            for i in range(len(line)-1):
                provi_X.append(line[i])
            provi_G = line[i+1]
            X.append(provi_X)
            G.append(provi_G)
        # Calculate F1
        for i in range(len(G)-1):
            d = []
            for j in range(i+1,len(G)):
                Delta_x = 0.0
                Delta_G = np.sqrt((G[i] - G[j])**2)
                for k in range(len(X[0])):
                    Delta_x = Delta_x + (X[i][k]-X[j][k])**2
                Delta_x = np.sqrt(Delta_x)
                if Delta_G > 0.0:
                    F1.append(Delta_x/Delta_G)
                    #F1.append(1/Delta_G)
        # Calculate F2
        for i in range(len(G)):
            for j in range(len(G)):
                if j==i+1:
                    dx = 0.0
                    for k in range(len(X[0])):
                        dx = dx + (X[i][k]-X[j][k])**2
                    dx = np.sqrt(dx)
                    F2.append(dx)
        # calculates average value for each SPF
        #av_f1 = sum(F1)/len(F1)
        av_f1 = statistics.mean(F1)
        stdev_f1 = statistics.stdev(F1)
        av_f2 = statistics.mean(F2)
        stdev_f2 = statistics.stdev(F2)
        new_S = factor_F1 * ((av_f1)**S_C1)*(len(G))**S_C2 * S_C3 **(len(X[0])) + 0.0
        new_a = factor_F2 * ((av_f2)**a_C1)*(len(G))**a_C2 * a_C3 **(len(X[0])) + 0.0

        average_F1.append(new_S)
        average_F2.append(new_a)
        error_F1.append(factor_F1 * abs(S_C1) * stdev_f1/abs(av_f1) * abs(new_S) * (len(G))**S_C2 * S_C3 **(len(X[0])))
        error_F2.append(factor_F2 * abs(a_C1) * stdev_f2/abs(av_f2) * abs(new_a) * (len(G))**a_C2 * a_C3 **(len(X[0])))

        #average_F2.append(av_f2)
        #error_F2.append(stdev_f2)

    # Add arrays with average of each landscape to a general array containing all 'a'
    final_S.append(average_F1)
    error_S.append(error_F1)
    final_a.append(average_F2)
    error_a.append(error_F2)

# Print final results
for i in range(len(adven)):
    #error = factor_F1 * 0.95 * 
    #print('### a = %i. Estimated S: %.2f +/- %.2f . Estimated a: %.2f +/- %.2f' % (adven[i],statistics.mean(final_S[i]), statistics.stdev(final_S[i]),statistics.mean(final_a[i]),statistics.stdev(final_a[i])))
    print('### a = %i. Estimated S: %.2f +/- %.2f . Estimated a: %.2f +/- %.2f' % (adven[i],statistics.mean(final_S[i]), statistics.mean(error_S[i]),statistics.mean(final_a[i]),statistics.mean(error_a[i])))

