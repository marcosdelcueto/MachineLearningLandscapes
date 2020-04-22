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
Nspf = 100
adven = [10,20,40,60,80,100]
######   END CUSTOMIZABLE PARAMETERS   ######
#################################################################################
for a in adven:
    average_F1 = []
    average_F2 = []
    for l in range(Nspf):
        # read data and assign it to X and G arrays
        f_data = open('%s_%i_%s.dat' % (data_file_name,a,l), 'r')
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
                    #F2.append((Delta_x)**(0.8)*len(G))
                    #print(i,j,Delta_x/Delta_G, Delta_x*len(G))
                #d.append(Delta_x)
            #print('test i, d:',i,d)
            #if i+1 == 1:          d0=statistics.mean(d)
            #if i+1 == 0.1*len(G): d1=statistics.mean(d)
            #if i+1 == 0.9*len(G): d2=statistics.mean(d)
            #if i+1 == len(G):     d3=statistics.mean(d)
                #print('Delta_x',Delta_x)
                #print('Delta_G',Delta_G)
        d0=[]
        d1=[]
        d2=[]
        d3=[]
        for i in range(len(G)):
            for j in range(len(G)):
                if j==i+1:
                    dx = 0.0
                    for k in range(len(X[0])):
                        dx = dx + (X[i][k]-X[j][k])**2
                    dx = np.sqrt(dx)
                    #print('d=',dx)
                    d0.append(dx)
            #if i+1 >=1 and i+1 <0.1*len(G):
                #for j in range(len(G)):
                    #if i!=j:
                        #dx = 0.0
                        #for k in range(len(X[0])):
                            #dx = dx+ (X[i][k]-X[j][k])**2
                        #dx = np.sqrt(dx)
                        #d0.append(dx)
            #if i+1 >=0.1 and i+1<0.2*len(G):
                #for j in range(len(G)):
                    #if i!=j:
                        #dx = 0.0
                        #for k in range(len(X[0])):
                            #dx = dx+ (X[i][k]-X[j][k])**2
                        #dx = np.sqrt(dx)
                        #d1.append(dx)
            #if i+1 >=0.8 and i+1<0.9*len(G):
                #for j in range(len(G)):
                    #if i!=j:
                        #dx = 0.0
                        #for k in range(len(X[0])):
                            #dx = dx+ (X[i][k]-X[j][k])**2
                        #dx = np.sqrt(dx)
                        #d2.append(dx)
            #if i+1 >=0.9*len(G) and i+1 <len(G):
                #for j in range(len(G)):
                    #if i!=j:
                        #dx = 0.0
                        #for k in range(len(X[0])):
                            #dx = dx+ (X[i][k]-X[j][k])**2
                        #dx = np.sqrt(dx)
                        #d3.append(dx)
        #print('d0:', d0)
        #print('d1:', d1)
        av_d0=sum(d0)/len(d0)
        #av_d1=sum(d1)/len(d1)
        #av_d2=sum(d2)/len(d2)
        #av_d3=sum(d3)/len(d3)
        #print('average d0,d1,d2,d3',av_d0,av_d1,av_d2,av_d3)
                    
        average_F1.append(sum(F1)/len(F1))
        average_F2.append((av_d0)**0.75)
        #average_F2.append(abs(av_d0-av_d1)/abs(av_d2-av_d3))
        #average_F2.append(sum(F2)/len(F2))
    # Print S results
    #print(average_F1)
    #print(sum(average_F1)/len(average_F1))
    print("################")
    print("####","a = ", a, "####")
    print('F1 Mean:',statistics.mean(average_F1), '. Stdev:',statistics.stdev(average_F1))
    print('F2 Mean:',statistics.mean(average_F2), '. Stdev:',statistics.stdev(average_F2))
    print('--------------')
    print('Estimated S:', statistics.mean(average_F1)/(2**(2*len(X[0])+1)), '+/-', statistics.stdev(average_F1)/(2**(2*len(X[0])+1)))
    print('Estimated a:', statistics.mean(average_F2)*15**len(X[0]), '+/-', statistics.stdev(average_F2)*15**len(X[0]))
