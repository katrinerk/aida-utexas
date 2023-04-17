root="/home/cc/data/out_salads_70k_Indexed"
echo $root
src=/home/cc/aida-utexas 
trainer="$src/pipeline/training/graph_salads/trainer.py"
name="base"
logdir="/home/cc/tensorboard/base"
indexer="$root/indexers.p"
PYTHONPATH=$src python $trainer --device 0 --data_dir $root --valid_dir "Val" --save_path "$root/model_$name" --indexer_info_file $indexer --hidden_size 300 --attention_size 300 --init_prob_force 0.25 --force_decay 0.9659 --weight_decay 0.001 --batch_size 25 --num_epochs 4 --force_every 3000 --log_dir $logdir --use_highest_ranked_gold --force --self_attend --attn_head_stmt_tail --valid_every 500
#\ --load_path "$root/model_1/gcn2-cuda_best_50000_0.ckpt"
