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

print_S=True
print_a=True


# Set value of general S factors
factor_F1 = 0.16 # 0.16
S_C1 =  0.55 # 0.55
S_C2 =  0.55 # 0.55
S_C3 =  1.00 # 1.00
# Set value of general a factors
factor_F2 = 500 # 232   # 500
a_C1 =  0.72  #0.70     # 0.72
a_C2 = -0.20 #-1.5      # -0.2
a_C3 =  1.00 # 1.0      # 1.0

# Loop for each adventurousness
counter_adven = 0
for a in adven:
    #print('########### NEW a ###########')
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
        F2_num = []
        F2_den = []
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
        if print_S==True:
            # Calculate F1
            ################### Option 1 
            #for j in range(i+1,len(G)):
            for i in range(len(G)):
                for j in range(len(G)):
                    if j>i:
                    #if j==i+1:
                    #if i==99:
                        Delta_x = 0.0
                        Delta_G = np.sqrt((G[i] - G[j])**2)
                        for k in range(len(X[0])):
                            Delta_x = Delta_x + (X[i][k]-X[j][k])**2
                        Delta_x = np.sqrt(Delta_x)
                        if Delta_G > 0.0:
                            F1.append(Delta_G/Delta_x**1.5)
                            #F1.append(Delta_G/Delta_x)
                        #F1.append(Delta_x)
            ###################
            ################### Option 2
            #for i in range(len(G)):
                #for j in range(len(G)):
                    #if j==i+1:
                        #Delta_x = 0.0
                        #Delta_G = np.sqrt((G[i] - G[j])**2)
                        #for k in range(len(X[0])):
                            #Delta_x = Delta_x + (X[i][k]-X[j][k])**2
                        #Delta_x = np.sqrt(Delta_x)
                        #if Delta_G > 0.0:
                            #F1.append((Delta_x**1.5/Delta_G))
                            #F1.append(Delta_x/Delta_G)
        if print_a == True:
            # Calculate F2
            ################### Option 1: average with next point
            for i in range(len(G)):
                for j in range(len(G)):
                    if j==i+1:
                        dx = 0.0
                        for k in range(len(X[0])):
                            dx = dx + (X[i][k]-X[j][k])**2
                        dx = np.sqrt(dx)
                        F2.append(dx)
            ################## Option 2: average of last10%/first10%
            #for i in range(len(G)):
                #if i<21:
                    #for j in range(0,20):
                        #if j != i:
                            #Delta_xi = 0.0
                            #for k in range(len(X[0])):
                                #Delta_xi = Delta_xi + (X[i][k]-X[j][k])**2
                            #Delta_xi = np.sqrt(Delta_xi)
                            #F2_den.append(Delta_xi)
                #if i>179:
                    #for j in range(180,200):
                        #if j != i:
                            #Delta_xf = 0.0
                            #for k in range(len(X[0])):
                                #Delta_xf = Delta_xf + (X[i][k]-X[j][k])**2
                            #Delta_xf = np.sqrt(Delta_xf)
                            #F2_num.append(Delta_xf)
            #for i in range(len(F2_num)):
                #F2.append(F2_num[i]/F2_den[i])
            ################### Option 3: 
            #for i in range(len(G)):
                #if i>90:
                    #for j in range(91,100):
                        #if j != i:
                            #Delta_x = 0.0
                            #for k in range(len(X[0])):
                                #Delta_x = Delta_x + (X[i][k]-X[j][k])**2
                            #Delta_x = np.sqrt(Delta_x)
                            #F2.append(Delta_x*1600)
        ###################
        # calculates average value for each SPF
        if print_S==True:
            av_f1 = statistics.mean(F1)
            stdev_f1 = statistics.stdev(F1)
            new_S = factor_F1 * ((av_f1)**S_C1)*(len(G))**S_C2 * S_C3 **(len(X[0])) + 0.0
        if print_a==True:
            av_f2 = statistics.mean(F2)
            stdev_f2 = statistics.stdev(F2)
            new_a = factor_F2 * ((av_f2)**a_C1)*(len(G))**a_C2 * a_C3 **(len(X[0])) + 0.0


        if print_S==True:
            average_F1.append(1.0/new_S)
            error_F1.append((abs(S_C1) * abs(av_f1)**(S_C1-1)*stdev_f1 *  factor_F1 * (len(G))**S_C2 * S_C3**(len(X[0])))/new_S**2)
        if print_a==True:
            average_F2.append(new_a)
            error_F2.append(abs(a_C1) * abs(av_f2)**(a_C1-1)*stdev_f2 *  factor_F2 * (len(G))**a_C2 * a_C3**(len(X[0])))



        #average_F2.append(av_f2)
        #error_F2.append(stdev_f2)

    # Add arrays with average of each landscape to a general array containing all 'a'
    if print_S==True:
        final_S.append(average_F1)
        error_S.append(error_F1)
    if print_a==True:
        final_a.append(average_F2)
        error_a.append(error_F2)

    if print_S==True and print_a == False:
        print('### a = %i. Estimated S: %.2f +/- %.2f' % (adven[counter_adven],statistics.mean(final_S[counter_adven]), statistics.mean(error_S[counter_adven])))
    elif print_a==True and print_S==False:
        print('### a = %i. Estimated a: %.2f +/- %.2f' % (adven[counter_adven],statistics.mean(final_a[counter_adven]), statistics.mean(error_a[counter_adven])))
    elif print_a==True and print_S==True:
        print('### a = %i. Estimated S: %.2f +/- %.2f . Estimated a: %.2f +/- %.2f' % (adven[counter_adven],statistics.mean(final_S[counter_adven]), statistics.mean(error_S[counter_adven]),statistics.mean(final_a[counter_adven]),statistics.mean(error_a[counter_adven])))

    counter_adven = counter_adven +1

# Print final results
#for i in range(len(adven)):
    #print('### a = %i. Estimated S: %.2f +/- %.2f . Estimated a: %.2f +/- %.2f' % (adven[i],statistics.mean(final_S[i]), statistics.mean(error_S[i]),statistics.mean(final_a[i]),statistics.mean(error_a[i])))

