root="/home/cc/data/out_salads_70k_Indexed_Geo"
echo $root
src=/home/cc/aida-utexas 
trainer="$src/pipeline/training/graph_salads/trainer.py"
name="geo_all"
logdir="/home/cc/tensorboard/$name"
indexer="$root/indexers_geo.p"
PYTHONPATH=$src python $trainer --device 1 --data_dir $root --valid_dir "Val" --save_path "$root/model_$name" --indexer_info_file $indexer --hidden_size 300 --attention_size 300 --init_prob_force 0.25 --force_decay 0.9659 --weight_decay 0.001 --batch_size 25 --num_epochs 4 --force_every 3000 --log_dir $logdir --use_highest_ranked_gold --force --self_attend --attn_head_stmt_tail --valid_every 500 
# --load_path "$root/model_geo_more/gcn2-cuda_best_9000_0_0.69.ckpt"
