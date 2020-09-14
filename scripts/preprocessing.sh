#!/bin/bash

echo
echo

set -xu

input_run_name=$1
input_kb_path=$2
working_dir=$3
num_hyps=$4
num_hops=$5
do_coref_compression=$6
optional_args=$7

echo
python -m pipeline.preprocessing.process_input \
    "$input_kb_path" \
    "${working_dir}/${input_run_name}.json" \
    -s "$SIN" \
    -q "${working_dir}/query_jsons" \
    "$optional_args"

echo
if $do_coref_compression; then
    python -m pipeline.preprocessing.compress_coref \
        "${working_dir}/${input_run_name}.json" \
        "${working_dir}/${input_run_name}_compressed.json" \
        "${working_dir}/${input_run_name}_log.json" \
        "$optional_args"
    graph_name="${input_run_name}_compressed.json"
else
    graph_name="${input_run_name}.json"
fi

echo
python -m pipeline.preprocessing.make_hypothesis_seeds \
    "${working_dir}/${graph_name}" \
    "${working_dir}/query_jsons" \
    "${working_dir}/cluster_seeds_raw" \
    --max_num_seeds "$num_hyps" \
    "$optional_args"

echo
python -m pipeline.preprocessing.rerank_hypothesis_seeds \
    "${working_dir}/${graph_name}" \
    "${working_dir}/cluster_seeds_raw" \
    "${working_dir}/cluster_seeds" \
    --max_num_seeds "$num_hyps" \
    "$optional_args"

for seed_file in "${working_dir}"/cluster_seeds/*.json; do
    echo
    seed_name=${seed_file##*/}
    seed_name=${seed_name%%_*}
    python -m pipeline.preprocessing.crop_subgraph \
        "${working_dir}/${graph_name}" \
        "${seed_file}" \
        "${working_dir}/subgraph/${seed_name}"/ \
        -n "${num_hops}" \
        "$optional_args"
done

echo
./pipeline/preprocessing/build_TA2_tdb.sh \
    "$input_kb_path" "${working_dir}/tdb_database" 5
