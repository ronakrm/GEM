#!/bin/bash
echo 'Running Script to sweep parameters for OT Speed Test...'

OUTPUT_FILE="results/ot_1d_results.csv"

startID=1
endID=10

run () {
	# echo $1 $2 $3 $4
	python compute.py --seed $1 \
					-n $2 \
					-d $3 \
					--model $4 \
					--outfile $OUTPUT_FILE
}

replicate() { 
	for runID in $(seq $startID 1 $endID)
		do
			run $runID $1 $2 $3 &
		done
}

for model in 'demd' 'lp_1d_bary' 'sink_1d_bary' 'cvx'
do
	for d in 2 5 10 20 50
	do
		for n in 2 5 10 20 50
		do
			replicate $n $d $model
		done
		wait
	done
done


