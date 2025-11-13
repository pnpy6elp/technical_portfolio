repeat=3
epoch=500

device=cuda:0

budget=100

dataset=corafull  # corafull_mixed

method=sage_cog # gg_mixed # gg_simple_revised

python train.py \
--seed $seed \
--repeat $repeat \
--cls-epoch $epoch \
--cgl-method $method \
--tim \
--data-dir ./data \
--result-path ./results \
--dataset-name $dataset \
--budget $budget \
--device $device \
--max_supernodes $budget \
--fixed_supernode_count $budget \
--top_k $budget \
--minority_ratio 0.5 \
--community_algo leiden ;

