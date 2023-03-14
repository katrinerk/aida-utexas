root="/home/cc/training_data/Final_Salads_50k/out_salads_Indexed"
echo $root
PYTHONPATH=/home/cc/aida-utexas python pipeline/training/graph_salads/trainer.py --mode "validate" --data_dir $root --valid_dir "Val" --save_path "/home/cc" --load_path "/home/cc/aida-utexas/resources/gcn2-cuda_best_5000_1.ckpt" --indexer_info_file "/home/cc/aida-utexas/resources/indexers.p"
