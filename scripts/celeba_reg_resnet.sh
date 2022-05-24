data='celeba'
batch_size=32
learning_rate=0.001
outfile='results/celeba_resnet.csv'
#model='CelebANet'
model='resnet18' # approx 1.6GB with batchsize 32

seed=0

run () {
	python run.py  \
			--dataset $data  \
			--model $model \
			--batch_size $batch_size \
			--regType $1 \
			--n_classes 1 \
			--nbins 10 \
			--nSens 2 \
			--epochs 10 \
			--lambda_reg $2 \
			--train_seed $3 \
			--learning_rate $learning_rate \
			--outfile $outfile
}

for lamb in 0.001 0.01 0.1 1.0 10.0 100.0 1000.0 10000.0
do
	for regtype in 'demd' 'dp' 'eo'
	do
		run $regtype $lamb $seed &
	done
	wait
done

run 'none' 0.0 $seed

# for lamb in 0.002 0.02 0.2 2.0 20.0 200.0 2000.0 20000.0
# 	for regtype in 'demd' 'dp' 'eo'
# 	do
# 		run $regtype $lamb $seed &
# 	done
# 	wait
# done

# for lamb in 0.005 0.05 0.5 5.0 50.0 500.0 5000.0 50000.0
# 	for regtype in 'demd' 'dp' 'eo'
# 	do
# 		run $regtype $lamb $seed &
# 	done
# 	wait
# done

# regtype='demd'
# for lamb in 0.001 0.01 0.1 1.0
# do
#  	run $regtype $lamb $seed &
# done
# wait

# for lamb in 0.002 0.02 0.2 2.0
# do
# 	run $regtype $lamb $seed &
# done
# wait

# for lamb in 0.005 0.05 0.5 5.0
# do
# 	run $regtype $lamb $seed &
# done
# wait

# # #for lamb in 0.1 0.11 0.12 0.13 0.14 0.15 
# # for lamb in 0.11 0.12 0.13 0.14 0.15 
# # do
# # 	run $regtype $lamb $seed &
# # done
# # wait

# # #for lamb in 0.16 0.17 0.18 0.19 0.2
# # for lamb in 0.16 0.17 0.18 0.19
# # do
# # 	run $regtype $lamb $seed &
# # done
# # wait

# regtype='dp'
# for lamb in 1.0 10.0 100.0 1000.0
# do
# 	run $regtype $lamb $seed &
# done
# wait
# for lamb in 2.0 20.0 200.0 2000.0
# do
#  	run $regtype $lamb $seed &
# done
# wait

# for lamb in 5.0 50.0 500.0 5000.0
# do
# 	run $regtype $lamb $seed &
# done
# wait

# regtype='eo'
# for lamb in 1.0 10.0 100.0 1000.0
# do
# 	run $regtype $lamb $seed &
# done
# wait
# for lamb in 2.0 20.0 200.0 2000.0
# do
#  	run $regtype $lamb $seed &
# done
# wait

# for lamb in 5.0 50.0 500.0 5000.0
# do
# 	run $regtype $lamb $seed &
# done
# wait
