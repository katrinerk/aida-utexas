#!/bin/bash

set -xu

kb_name=$1
num_hyps=$2
num_hops=$3
need_coref_compression=$4

wd="data/${kb_name}"

python -m pipeline.preprocessing.process_input \
    "data/TA2/${kb_name}.ttl" \
    "data/SIN" \
    "${wd}/${kb_name}.json" \
    "${wd}/query_jsons"

if [[ "${need_coref_compression}" == 1 ]]
then
    python -m pipeline.preprocessing.compress_coref \
        "${wd}/${kb_name}.json" \
        "${wd}/${kb_name}_compressed.json" \
        "${wd}/${kb_name}_log.json"
    graph_name="${kb_name}_compressed.json"
else
    graph_name="${kb_name}.json"
fi

python -m pipeline.preprocessing.make_cluster_seeds \
    "${wd}/${graph_name}" \
    "${wd}/query_jsons" \
    "${wd}/cluster_seeds" \
    --max_num_seeds "${num_hyps}" \
    --early_cutoff 100

for seed_file in "${wd}"/cluster_seeds/*.json
do
    seed_name=${seed_file##*/}
    seed_name=${seed_name%%_*}
    python -m pipeline.preprocessing.crop_subgraph \
        "${wd}/${graph_name}" \
        "${seed_file}" \
        "${wd}/subgraph/${seed_name}"/ \
        -n "${num_hops}"
done

./pipeline/preprocessing/build_TA2_tdb.sh \
    "data/TA2/${kb_name}.ttl" "${wd}/tdb_database" 5
