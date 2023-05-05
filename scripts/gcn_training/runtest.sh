root="/home/cc/data/out_salads_70k_Indexed_test2"
src=/home/cc/aida-utexas 
trainer="$src/pipeline/training/graph_salads/trainer.py"
model="/home/cc/data/out_salads_70k_Indexed/model_base/gcn2-cuda_best_22000_0_0.71.ckpt"
PYTHONPATH=/home/cc/aida-utexas python $trainer --device 1 --mode "validate" --data_dir $root --valid_dir "Val" --save_path "/home/cc" --load_path $model --indexer_info_file "$root/indexers.p" --hidden_size 300 --attention_size 300 --init_prob_force 0.25 --force_decay 0.9659 --weight_decay 0.001 --batch_size 25 --num_epochs 4 --force_every 3000 --use_highest_ranked_gold --force --self_attend --attn_head_stmt_tail --valid_every 500
