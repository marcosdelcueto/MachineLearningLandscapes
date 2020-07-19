#!/bin/bash
t0=15
t1=200
t2=15

number_lines=$((${t0}+${t1}+${t2}))
remove_head=$((-1*${t0}))
remove_tail=$((${t2}+1))
for l in `seq 0 99`
do 
	counter_a=1
	for a in `echo "10 20 30 40 50 60 70 80 90 100"`
	do
		lines=$((${number_lines}*${counter_a}))
		grep "timestep " log_grid_l_${l}.log | head -n ${lines} | tail -n ${number_lines} | awk '{ for(i=3; i<NF; i++) printf "%s",$i OFS; if(NF) printf "%s",$NF; printf ORS}' | tail -n +${remove_tail} | head -n ${remove_head} > data_x_G_${a}_${l}.dat
		counter_a=$((${counter_a}+1))
	done
done
