#!/usr/bin/env python3
# Marcos del Cueto
# Department of Chemistry and MIF, University of Liverpool
#################################################################################
import ast
import numpy as np
#################################################################################
######   START CUSTOMIZABLE PARAMETERS ########
data_file_name = 'data_x_G'      # name of input file
Ndata_files = 18
######   END CUSTOMIZABLE PARAMETERS   ######
#################################################################################
average_F1 = []
for l in range(Ndata_files):
    # read data and assign it to X and G arrays
    f_data = open('%s_%s.dat' % (data_file_name,l), 'r')
    X = []
    G = []
    F1 = []
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
            #print(i,j)
            #print('Delta_x',Delta_x)
            #print('Delta_G',Delta_G)
    average_F1.append(sum(F1)/len(F1))
print(average_F1)
print(sum(average_F1)/len(average_F1))
    #print('-----------')
    #print(F1)
    #print(sum(F1)/len(F1))
        #print(line)
        #print(type(line))
        #print('----------')
        #print(new_line)
        #print(type(new_line))
        #print('----------')
