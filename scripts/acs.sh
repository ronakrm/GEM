data='acs-income'
batch_size=256
learning_rate=0.0001
outfile='results/acs_inc_comp.csv'


run () {
	python run.py  \
			--dataset $data  \
			--model $3 \
			--batch_size $batch_size \
			--input_size 10 \
			--regType $2 \
			--n_classes 1 \
			--nbins 2 \
			--nSens 9 \
			--epochs 10 \
			--lambda_reg $1 \
			--learning_rate $learning_rate \
			--outfile $outfile
}

# run 0.0 'dp' 'ACSRegressor'

reg='demd'
for model in 'ACSRegressor' 'ACSNet' #'ACSDeepNet'
do
	for lamb in 0.1 0.2 0.3 0.4
	do
		run $lamb $reg $model &
	done
	wait
	for lamb in 0.5 0.6 0.7 0.8 0.9
	do

		run $lamb $reg $model &
	done
	wait
done

# for lamb in 0.0 1 0.1 10 0.01 100
# do
# 	for model in 'ACSRegressor' 'ACSNet' #'ACSDeepNet'
# 	do
# 		for reg in 'dp' 'eo' 'demd'
# 		do
# 			run $lamb $reg $model &
# 		done
# 	done
# 	wait
# done