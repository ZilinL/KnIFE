dataset='OpenBMI-6domain'
algorithm=('ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'VREx' 'GroupDRO' 'MLDG' 'ANDMask' 'Knife')
test_envs=6
gpu_ids=7
data_dir='./data/OpenBMI/filteredMat/'
max_epoch=200
net='EEGNet'
output='./train_output/test'
batch_size=32


# ERM 
python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --output $output --batch_size $batch_size \
--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[0]} --gpu_id $gpu_ids

# DANN 
python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --output $output --batch_size $batch_size \
--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[1]} --gpu_id $gpu_ids

# RSC 
python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --output $output --batch_size $batch_size \
--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[2]} --gpu_id $gpu_ids

# Mixup 
python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --output $output --batch_size $batch_size \
--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[3]} --gpu_id $gpu_ids

# MMD 
python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --output $output --batch_size $batch_size \
--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[4]} --gpu_id $gpu_ids

# CORAL 
python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --output $output --batch_size $batch_size \
--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[5]} --gpu_id $gpu_ids

# VREx 
python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --output $output --batch_size $batch_size \
--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[6]} --gpu_id $gpu_ids

# GroupDRO 
python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --output $output --batch_size $batch_size \
--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[7]} --gpu_id $gpu_ids

# MLDG 
python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --output $output --batch_size $batch_size \
--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[8]} --gpu_id $gpu_ids

# ANDMask 
python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --output $output --batch_size $batch_size \
--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[9]} 

# Knife
python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --output $output --batch_size $batch_size \
--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[10]} --gpu_id $gpu_ids
