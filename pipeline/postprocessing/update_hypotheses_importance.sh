#!/usr/bin/env bash

set -u

tdb_dir=$1
query_dir=$2
count=$3

for i in $(seq -f "%03g" 1 "${count}")
do
  echo "Adding importance values to hypothesis-${i}"
  for f in "${query_dir}"/hypothesis-"${i}"*
  do
    tdbupdate --loc "${tdb_dir}"/hypothesis-"${i}" --update "${f}"
  done &
done

wait
