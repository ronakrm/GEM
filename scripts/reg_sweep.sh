data='acs-income'
batch_size=256
learning_rate=0.0001
outfile='results/sweep_res.csv'
model='ACSRegressor'
regtype='demd'

run () {
	python run.py  \
			--dataset $data  \
			--model $model \
			--batch_size $batch_size \
			--input_size 10 \
			--regType $regtype \
			--n_classes 1 \
			--nbins 10 \
			--nSens 9 \
			--epochs 10 \
			--lambda_reg $1 \
			--train_seed $2 \
			--learning_rate $learning_rate \
			--outfile $outfile
}

seed=0
# for seed in 0 1 2 3 4
# do
#for lamb in 0.01 0.02 0.03 0.04 0.05
for lamb in 0 0.06 0.07 0.08 0.09 0.1
do
	run $lamb $seed &
done
wait
# done
