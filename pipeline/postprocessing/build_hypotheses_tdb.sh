#!/usr/bin/env bash

set -xu

input_dir=$1
output_dir=$2
count=$3

mkdir -p "${output_dir}"

for i in $(seq -f "%03g" 1 "${count}")
do
    hyp_file="${input_dir}"/hypothesis-"${i}"-raw.ttl
    if [ ! -f "$hyp_file" ]; then
        continue
    fi
    hyp_output_dir="${output_dir}"/hypothesis-"${i}"
    if [ -d "$hyp_output_dir" ]; then
        rm -rf "$hyp_output_dir"
    fi
    mkdir -p "$hyp_output_dir"
    tdbloader2 --loc "$hyp_output_dir" --jvm-args -Xmx10g "$hyp_file"
done
