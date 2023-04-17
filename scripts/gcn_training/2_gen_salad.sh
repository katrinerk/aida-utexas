root="/home/cc/aida-utexas/pipeline/data_gen/graph_salads"
PYTHONPATH="/home/cc/aida-utexas" python "$root/gen_salads.py" --single_doc_graphs_folder "/home/cc/data/test_file_gen" --data_size 70000 --perc_train 0.82 --perc_test 0.14 --out_data_dir "/home/cc/out_salad" --match_event_names --match_entity_names
