#!/bin/bash

set -xu

kb_name=$1
sin_name=$2
num_hyps=$3
sin_id=$4
frame_id=$5
run_id=$6
need_coref_compression=$7

wd=data/"${kb_name}"

if [[ "${need_coref_compression}" == 1 ]]
then
    python -m pipeline.postprocessing.posthoc_filter_hypotheses \
        "${wd}/${kb_name}_compressed.json" \
        "${wd}/result_jsons" \
        "${wd}/result_jsons_filtered"

    python -m pipeline.postprocessing.recover_coref \
        "${wd}/result_jsons_filtered/" \
        "${wd}/result_jsons_recovered" \
        "${wd}/${kb_name}.json" \
        "${wd}/${kb_name}_compressed.json" \
        "${wd}/${kb_name}_log.json"

    results_dir_name="result_jsons_recovered"
else
    python -m pipeline.postprocessing.posthoc_filter_hypotheses \
        "${wd}/${kb_name}.json" \
        "${wd}/result_jsons" \
        "${wd}/result_jsons_filtered"

    results_dir_name="result_jsons_filtered"
fi

python -m pipeline.postprocessing.query_hypotheses \
    "${wd}/${kb_name}.json" \
    "${wd}/${results_dir_name}/${sin_name}.json" \
    "${wd}/tdb_database" \
    "${wd}/hypotheses/${sin_name}/raw" \
    --top "${num_hyps}"

python -m pipeline.postprocessing.add_importance_value_w_sparql_update \
    "${wd}/${kb_name}.json" \
    "${wd}/${results_dir_name}/${sin_name}.json" \
    "${wd}/hypotheses/${sin_name}/update_importance" \
    "${sin_id}_${frame_id}"

python -m pipeline.postprocessing.add_handle_w_sparql_update \
    "${wd}/${kb_name}.json" \
    "${wd}/${results_dir_name}/${sin_name}.json" \
    "${wd}/hypotheses/${sin_name}/update_handle"

./pipeline/postprocessing/build_hypotheses_tdb.sh \
    "${wd}/hypotheses/${sin_name}/raw" \
    "${wd}/hypotheses/${sin_name}/tdb" \
    "${num_hyps}"

./pipeline/postprocessing/update_hypotheses_importance.sh \
    "${wd}/hypotheses/${sin_name}/tdb" \
    "${wd}/hypotheses/${sin_name}/update_importance" \
    "${num_hyps}"

./pipeline/postprocessing/update_hypotheses_handle.sh \
    "${wd}/hypotheses/${sin_name}/tdb" \
    "${wd}/hypotheses/${sin_name}/update_handle" \
    "${num_hyps}"

./pipeline/postprocessing/dump_hypotheses_tdb.sh \
    "${wd}/hypotheses/${sin_name}/tdb" \
    "${wd}/hypotheses/${sin_name}/final" \
    "${run_id}" "${sin_id}" "${num_hyps}"

rm -rf "${wd}/hypotheses/${sin_name}/raw" \
    "${wd}/hypotheses/${sin_name}/update_importance" \
    "${wd}/hypotheses/${sin_name}/update_handle" \
    "${wd}/hypotheses/${sin_name}/tdb"
