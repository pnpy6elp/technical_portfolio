repeat=3
epoch=500

device=cuda:0

budget=100

dataset=corafull  

method=sage_cog 

python ../src/train.py \
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


repeat=3
epoch=500

device=cuda:0

budget=200

dataset=arxiv  
method=sage_cog 

python ../src/train.py \
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

repeat=3
epoch=500

device=cuda:0

budget=300

dataset=reddit

method=sage_cog 

python ../src/train.py \
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

repeat=3
epoch=500

device=cuda:0

budget=400

dataset=products

method=sage_cog

python ../src/train.py \
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

