#!/bin/bash

echo
echo

set -xu

input_run_name=$1
run_name=$2
output_dir=$3
working_dir=$4
num_hyps=$5
do_coref_compression=$6
sin_id_prefix=$7
optional_args=$8

#sin_name=$2
#num_hyps=$3
#sin_id=$4
#frame_id=$5
#run_id=$6
#need_coref_compression=$7

echo
if "$do_coref_compression"; then
    python -m pipeline.postprocessing.posthoc_filter_hypotheses \
        "${working_dir}/${input_run_name}_compressed.json" \
        "${working_dir}/result_jsons" \
        "${working_dir}/result_jsons_filtered" \
        "$optional_args"

    python -m pipeline.postprocessing.recover_coref \
        "${working_dir}/result_jsons_filtered/" \
        "${working_dir}/result_jsons_recovered" \
        "${working_dir}/${input_run_name}.json" \
        "${working_dir}/${input_run_name}_compressed.json" \
        "${working_dir}/${input_run_name}_log.json" \
        "$optional_args"

    results_dir="result_jsons_recovered"
else
    python -m pipeline.postprocessing.posthoc_filter_hypotheses \
        "${working_dir}/${input_run_name}.json" \
        "${working_dir}/result_jsons" \
        "${working_dir}/result_jsons_filtered" \
        "$optional_args"

    results_dir="result_jsons_filtered"
fi

for result_file in "$working_dir"/"$results_dir"/*.json; do
    echo
    sin_name=${result_file##*/}
    sin_name=${sin_name%%.*}

    python -m pipeline.postprocessing.query_hypotheses \
        "${working_dir}/${input_run_name}.json" \
        "${result_file}" \
        "${working_dir}/tdb_database" \
        "${working_dir}/hypotheses/${sin_name}/raw" \
        --top "${num_hyps}" \
        "$optional_args"

    ./pipeline/postprocessing/build_hypotheses_tdb.sh \
        "${working_dir}/hypotheses/${sin_name}/raw" \
        "${working_dir}/hypotheses/${sin_name}/tdb" \
        "${num_hyps}"

    python -m pipeline.postprocessing.add_importance_value_w_sparql_update \
        "${working_dir}/${input_run_name}.json" \
        "${result_file}" \
        "${working_dir}/hypotheses/${sin_name}/update_importance" \
        "${sin_id_prefix}_${sin_name}_F1" \
        "$optional_args"

    python -m pipeline.postprocessing.add_handle_w_sparql_update \
        "${working_dir}/${input_run_name}.json" \
        "${result_file}" \
        "${working_dir}/hypotheses/${sin_name}/update_handle" \
        "$optional_args"

    ./pipeline/postprocessing/update_hypotheses_importance.sh \
        "${working_dir}/hypotheses/${sin_name}/tdb" \
        "${working_dir}/hypotheses/${sin_name}/update_importance" \
        "${num_hyps}"

    ./pipeline/postprocessing/update_hypotheses_handle.sh \
        "${working_dir}/hypotheses/${sin_name}/tdb" \
        "${working_dir}/hypotheses/${sin_name}/update_handle" \
        "${num_hyps}"

    ./pipeline/postprocessing/dump_hypotheses_tdb.sh \
        "${working_dir}/hypotheses/${sin_name}/tdb" \
        "${output_dir}" \
        "${input_run_name}.${run_name}" \
        "${sin_id_prefix}_${sin_name}" \
        "${num_hyps}"

#    rm -rf "${working_dir}/hypotheses/${sin_name}/raw" \
#        "${working_dir}/hypotheses/${sin_name}/update_importance" \
#        "${working_dir}/hypotheses/${sin_name}/update_handle" \
#        "${working_dir}/hypotheses/${sin_name}/tdb"

done
