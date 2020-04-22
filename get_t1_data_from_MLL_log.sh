#!/bin/bash


t1=120
t0=12
data_file_name='data_x_G' 
Nspf=100
adv=(10 20 40 60 80 100)

for l in `seq 0 $((Nspf-1))`
do
   counter=0
   for i in `seq ${#adv[@]}`
   do 
		a=${adv[counter]}
      #echo ${counter} ${i} ${l} ${a}
    	grep "timestep " log_grid_l_${l}.log | awk '$2>='"${t0}"'{print $0}' | awk 'NR>'"${t1}"'*'"${counter}"'&&NR<='"${t1}"'*'"${counter}"'+'"${t1}"' {print $3,$4,$5}' > ${data_file_name}_${a}_${l}.dat
      counter=$((counter+1))
   done
done
