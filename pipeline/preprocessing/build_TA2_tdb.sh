#!/usr/bin/env bash

set -xu

ttl_path=$1
tdb_dir=$2
num_copies=$3

mkdir -p "${tdb_dir}"/copy_0
tdbloader2 --loc "${tdb_dir}"/copy_0 --jvm-args -Xmx40g "${ttl_path}"

for ((i=1;i<num_copies;i++))
do
  cp -r "${tdb_dir}"/copy_0/ "${tdb_dir}"/copy_${i}/
done
