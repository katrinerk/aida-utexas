#!/usr/bin/env bash

set -u

tdb_dir=$1
query_dir=$2
count=$3

for i in $(seq -f "%03g" 1 "${count}")
do
    tdb_path="${tdb_dir}"/hypothesis-"${i}"
    sparql_file="${query_dir}"/hypothesis-"${i}"-update.rq
    if [ ! -d "$tdb_path" ] || [ ! -f "$sparql_file" ]; then
        continue
    fi
    echo "Adding handles to hypothesis-${i}"
    tdbupdate --loc "${tdb_path}" --update "${sparql_file}"
done

wait
