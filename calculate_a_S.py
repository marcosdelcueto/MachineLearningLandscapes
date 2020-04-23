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
adven = [10,20,40,60,80,100]     # array with adventurousness values
######   END CUSTOMIZABLE PARAMETERS   ######
#################################################################################
final_S = []
final_a = []
# Set value of general factors
factor_F1 = 0.27
factor_F2 = 120
# Loop for each adventurousness
for a in adven:
    # initialize final arrays
    average_F1 = []
    average_F2 = []
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
        av_f1 = sum(F1)/len(F1)
        av_f2 = (sum(F2)/len(F2))
        average_F1.append(((av_f1)**0.95)/((len(G))**0.50) * 1 **(len(X[0])) + 0.0)
        average_F2.append(((av_f2)**0.78)/((len(G))**0.55) * 1 **(len(X[0])) + 0.0)

    # Add arrays with average of each landscape to a general array containing all 'a'
    final_S.append(average_F1)
    final_a.append(average_F2)

# Print final results
for i in range(len(adven)):
    print('### a = %i. Estimated S: %.2f +/- %.2f . Estimated a: %.2f +/- %.2f' % (adven[i],statistics.mean(final_S[i])*factor_F1, statistics.stdev(final_S[i])*factor_F1,statistics.mean(final_a[i])*factor_F2,statistics.stdev(final_a[i])*factor_F2))

