#!/usr/bin/env bash

set -u

tdb_dir=$1
query_dir=$2
count=$3

for i in $(seq -f "%03g" 1 "${count}")
do
  echo "Adding handles to hypothesis-${i}"
  tdbupdate --loc "${tdb_dir}"/hypothesis-"${i}" --update "${query_dir}"/hypothesis-"${i}"-update.rq
done

wait
