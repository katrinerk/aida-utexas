#!/usr/bin/env bash

set -xu

ttl_path=$1
tdb_dir=$2
num_copies=$3

if [ -d "${tdb_dir}"/copy_0 ]; then
    rm -rf "${tdb_dir}"/copy_0
fi
mkdir -p "${tdb_dir}"/copy_0
tdbloader2 --loc "${tdb_dir}"/copy_0 --jvm-args -Xmx40g "${ttl_path}"

for ((i=1;i<num_copies;i++))
do
    if [ -d "${tdb_dir}"/copy_${i} ]; then
        rm -rf "${tdb_dir}"/copy_${i}
    fi
    cp -r "${tdb_dir}"/copy_0/ "${tdb_dir}"/copy_${i}/
done
