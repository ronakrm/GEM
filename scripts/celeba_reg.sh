data='celeba'
batch_size=64
learning_rate=0.001
outfile='results/celeba_results_new.csv'
model='CelebANet'

seed=123

run () {
	python run.py  \
			--dataset $data  \
			--model $model \
			--batch_size $batch_size \
			--input_size 10 \
			--regType $1 \
			--n_classes 1 \
			--nbins 10 \
			--nSens 2 \
			--epochs 25 \
			--lambda_reg $2 \
			--train_seed $3 \
			--learning_rate $learning_rate \
			--outfile $outfile
}


regtype='demd'
for lamb in 0 0.001 0.01 0.1 1.0
do
	run $regtype $lamb $seed &
done
wait

for lamb in 0.002 0.02 0.2
do
	run $regtype $lamb $seed &
done
wait

for lamb in 0.005 0.05 0.5
do
	run $regtype $lamb $seed &
done
wait

# #for lamb in 0.1 0.11 0.12 0.13 0.14 0.15 
# for lamb in 0.11 0.12 0.13 0.14 0.15 
# do
# 	run $regtype $lamb $seed &
# done
# wait

# #for lamb in 0.16 0.17 0.18 0.19 0.2
# for lamb in 0.16 0.17 0.18 0.19
# do
# 	run $regtype $lamb $seed &
# done
# wait

regtype='dp'
for lamb in 0 0.001 0.01 0.1 1.0 10.0
do
	run $regtype $lamb $seed &
done
wait

for lamb in 0.002 0.02 0.02 0.2 2.0
do
	run $regtype $lamb $seed &
done
wait

for lamb in 0.005 0.05 0.5 5.0
do
	run $regtype $lamb $seed &
done
wait

regtype='eo'
for lamb in 0 0.001 0.01 0.1 1.0 10.0
do
	run $regtype $lamb $seed &
done
wait

for lamb in 0.002 0.02 0.02 0.2 2.0
do
	run $regtype $lamb $seed &
done
wait

for lamb in 0.005 0.05 0.5 5.0
do
	run $regtype $lamb $seed &
done
wait


regtype='dp'
for lamb in 100 1000 10000
do
	run $regtype $lamb $seed &
done
wait

for lamb in 20 200 2000 20000
do
	run $regtype $lamb $seed &
done
wait

for lamb in 50 500 5000 50000
do
	run $regtype $lamb $seed &
done
wait

regtype='eo'
for lamb in 100 1000 10000
do
	run $regtype $lamb $seed &
done
wait

for lamb in 20 200 2000 20000
do
	run $regtype $lamb $seed &
done
wait

for lamb in 50 500 5000 50000
do
	run $regtype $lamb $seed &
done
wait