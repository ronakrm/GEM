# thresh.sh

data='acs-employ'
batch_size=512
learning_rate=0.001
outfile='results/thresh_sweep_full.csv'
model='ACSRegressor'
#model='ACSDeepNet'

run () {
	python thresholder.py  \
			--dataset $data  \
			--model $model \
			--batch_size $batch_size \
			--input_size 16 \
			--train_threshold 0.5 \
			--regType $1 \
			--n_classes 1 \
			--nbins 10 \
			--nSens 9 \
			--epochs 10 \
			--lambda_reg $2 \
			--learning_rate $learning_rate \
			--outfile $outfile
}


run 'none' 0 &
run 'dp' 100 &
run 'eo' 100 &
run 'demd' 0.05 &
wait