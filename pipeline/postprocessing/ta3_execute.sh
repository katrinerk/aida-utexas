#!/bin/bash

#root = direct/path/to/pipeline
cd /Users/cookie/workspace/aida-utexas/pipeline/postprocessing

chmod u+x ta3_execute.sh 

###### user needs to give following information
ttl_file_dir=direct/path/to/ttl/file/dir #type in

ttl_file=ttl_file_name #type in

pre_output=direct/path/to/target/output/dir #type in

query_path=None #type in, defualt value = None

query_output=None #type in, defualt value = None

########conducting preprocessing process
echo "Conducting preprocessing of ta3"
cd ..
cd preprocessing
if [ $query_path == None ]
then 
    python3 preprocess_claims_relativepath.py $ttl_file_dir $pre_output
else
    python3 preprocess_claims_relativepath.py $ttl_file_dir $pre_output -q $query_path -Q $query_output
fi

########conducting working process
echo "Conducting main processing of ta3"



########conducting postprocessing process
echo "Conducting postprocessing of ta3"
cd ..
cd postprocessing
#assume working directory is under the pre_output dir
working_dir="$pre_output/Working"
run_id=runname #type in
kb_path="$ttl_file_dir/$ttl_file"
graph_path="$pre_output/json/$ttl_file.json"

echo "Please enter the conditions you want to test on e.g. Condition5/6/7, or all"
read condition 

output_dir="$pre_output/Out"

if [$condition == Condition5]
then 
    python3 produce_claim_aif_c5_addedge_newest.py $working_dir $run_id Condition5 $graph_path $kb_path $output_dir
else
    if [$condition != all]
    then 
        python3 produce_claim_aif_c6_addedge_newest.py $working_dir $run_id $condition $graph_path $kb_path $output_dir
    else
        python3 produce_claim_aif_c5_addedge_newest.py $working_dir $run_id Condition5 $graph_path $kb_path $output_dir
        python3 produce_claim_aif_c6_addedge_newest.py $working_dir $run_id Condition6 $graph_path $kb_path $output_dir
        python3 produce_claim_aif_c6_addedge_newest.py $working_dir $run_id Condition7 $graph_path $kb_path $output_dir
    fi
fi