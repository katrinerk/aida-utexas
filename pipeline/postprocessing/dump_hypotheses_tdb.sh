#!/usr/bin/env bash

set -u

tdb_dir=$1
output_dir=$2
run_id=$3
soin_id=$4
count=$5

mkdir -p "${output_dir}"

for i in $(seq -f "%03g" 1 "${count}")
do
    tdb_path="${tdb_dir}"/hypothesis-"${i}"
    if [ ! -d "$tdb_path" ]; then
        continue
    fi
    output_path=${output_dir}/${run_id}.${soin_id}.${soin_id}_F1.H${i}.ttl
    echo "Dumping hypothesis-${i} to ${output_path}"
    tdbdump --formatted trig --loc "${tdb_path}" > "${output_path}"
done
