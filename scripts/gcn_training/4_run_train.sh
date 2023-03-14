root="/home/cc/data/out_salads_70k_Indexed"
echo $root
src=/home/cc/aida-utexas 
trainer="$src/pipeline/training/graph_salads/trainer.py"
PYTHONPATH=$src python $trainer --data_dir $root --valid_dir "Val" --save_path "$root/model" --indexer_info_file "$root/indexers.p" --init_prob_force 0.25 --force_decay 0.9659 --weight_decay 0.001 --batch_size 25 --num_epochs 4 --force_every 3000 --learning_rate 0.00001 --num_layers 1 --use_highest_ranked_gold --force --self_attend --attn_head_stmt_tail
#\ --device 1 --load_path "$root/model_1/gcn2-cuda_best_50000_0.ckpt"
