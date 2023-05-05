root="/home/cc/data/out_salads_70k_Indexed_GeoPerson_test2"
indexer="/home/cc/data/out_salads_70k_Indexed_GeoPerson/indexers_geoper_allwiki.p"
src=/home/cc/aida-utexas 
trainer="$src/pipeline/training/graph_salads/trainer.py"
model="/home/cc/data/out_salads_70k_Indexed_GeoPerson/model_geo_all_per_all_wiki/gcn2-cuda_best_25000_0_0.73.ckpt"
PYTHONPATH=/home/cc/aida-utexas python $trainer --device 1 --mode "validate" --data_dir $root --valid_dir "Val" --save_path "/home/cc" --load_path $model --indexer_info_file $indexer --hidden_size 302 --attention_size 302 --init_prob_force 0.25 --force_decay 0.9659 --weight_decay 0.001 --batch_size 25 --num_epochs 4 --force_every 3000 --use_highest_ranked_gold --force --self_attend --attn_head_stmt_tail --valid_every 500
