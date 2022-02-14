# thresh.sh

data='acs-income'
batch_size=256
learning_rate=0.0001
outfile='results/AA_invary_tight.csv'
model='ACSNet'

run () {
	python thresholder.py  \
			--dataset $data  \
			--model $model \
			--batch_size $batch_size \
			--input_size 10 \
			--regType $1 \
			--n_classes 1 \
			--nbins 10 \
			--nSens 9 \
			--epochs 10 \
			--lambda_reg $2 \
			--learning_rate $learning_rate \
			--outfile $outfile
}

run 'dp' 100 &
run 'eo' 100 &
run 'demd' 0.05 &
wait