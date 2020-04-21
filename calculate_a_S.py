#!/usr/bin/env python3
# Marcos del Cueto
# Department of Chemistry and MIF, University of Liverpool
#################################################################################
import ast
import numpy as np
import statistics
#################################################################################
######   START CUSTOMIZABLE PARAMETERS ########
data_file_name = 'data_x_G'      # name of input file
Ndata_files = 18
######   END CUSTOMIZABLE PARAMETERS   ######
#################################################################################
average_F1 = []
average_F2 = []
for l in range(Ndata_files):
    # read data and assign it to X and G arrays
    f_data = open('%s_%s.dat' % (data_file_name,l), 'r')
    X = []
    G = []
    F1 = []
    F2 = []
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
    for i in range(len(G)):
        for j in range(i+1,len(G)):
            Delta_x = 0.0
            Delta_G = np.sqrt((G[i] - G[j])**2)
            for k in range(len(X[0])):
                Delta_x = Delta_x + (X[i][k]-X[j][k])**2
            Delta_x = np.sqrt(Delta_x)
            F1.append(Delta_x/Delta_G)
            F2.append(Delta_x*len(G))
            #print(i,j)
            #print('Delta_x',Delta_x)
            #print('Delta_G',Delta_G)
    average_F1.append(sum(F1)/len(F1))
    average_F2.append(sum(F2)/len(F2))
# Print S results
#print(average_F1)
#print(sum(average_F1)/len(average_F1))
print('F1 Mean:',statistics.mean(average_F1), '. Stdev:',statistics.stdev(average_F1))
print('F2 Mean:',statistics.mean(average_F2), '. Stdev:',statistics.stdev(average_F2))
print('--------------')
print('Estimated S:', statistics.mean(average_F1)/32, '+/-', statistics.stdev(average_F1)/32)
print('Estimated a:', statistics.mean(average_F2)*4, '+/-', statistics.stdev(average_F2)*4)
