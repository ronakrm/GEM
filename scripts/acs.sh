data='acs-income'
batch_size=256
outfile='results/acs_inc_results.csv'


run () {
	python run.py  \
			--dataset $data  \
			--model $model \
			--batch_size $batch_size \
			--input_size 10 \
			--n_classes 1 \
			--epochs 5 \
			--lambda_reg $1 \\
			--outfile $outfile
}

model='ACSRegressor'
run 0.0
run 0.0001
run 0.001
run 0.01
run 0.1

model='ACSNet'
run 0.0
run 0.0001
run 0.001
run 0.01
run 0.1
