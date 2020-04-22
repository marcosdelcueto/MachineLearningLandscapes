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
        #print('X:')
        #print(X)
        #print('G:')
        #print(G)
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
                    d0.append(dx)
        av_d0=sum(d0)/len(d0)
        #print('average d0',av_d0)
                    
        average_F1.append(sum(F1)/len(F1))
        average_F2.append((av_d0)**0.75)
    # Print S results
    factor_F1 = 2**(2*len(X[0])+1)
    factor_F2 = 15**len(X[0])
    #print(average_F1)
    #print(sum(average_F1)/len(average_F1))
    #print(average_F2)
    #print(sum(average_F2)/len(average_F2))
    print("################")
    print("####","a = ", a, "####")
    print('F1 Mean:',statistics.mean(average_F1), '. Stdev:',statistics.stdev(average_F1))
    print('F2 Mean:',statistics.mean(average_F2), '. Stdev:',statistics.stdev(average_F2))
    print('--------------')
    print('Estimated S:', statistics.mean(average_F1)/factor_F1, '+/-', statistics.stdev(average_F1)/factor_F1)
    print('Estimated a:', statistics.mean(average_F2)*factor_F2, '+/-', statistics.stdev(average_F2)*factor_F2)
