#!/usr/bin/env bash

set -u

input_dir=$1
output_dir=$2
count=$3

mkdir "${output_dir}"

for i in $(seq -f "%03g" 1 "${count}")
do
  mkdir -p "${output_dir}"/hypothesis-"${i}"
  tdbloader2 --loc "${output_dir}"/hypothesis-"${i}" --jvm-args -Xmx10g "${input_dir}"/hypothesis-"${i}"-raw.ttl
done
