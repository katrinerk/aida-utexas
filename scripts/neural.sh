#!/bin/bash

echo
echo

working_dir=$1
device=$2
optional_args=$3

download_from_gdrive() {
    file_id=$1
    file_name=$2
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=${file_id}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${file_id}" -O "$file_name" && rm -rf /tmp/cookies.txt
}

indexer_download_id="1lw2ykhEKTutEYvw9qKzjIIygBwUUWS45"
indexer_path="resources/indexers.p"

model_download_id="1uWbppTp1TE4YvX3euVDpH9PnyLp7_sHG"
model_path="resources/gcn2-cuda_best_5000_1.ckpt"

set -x

if [ ! -e "$indexer_path" ]; then
    printf "\n\nDownloading indexer file from Google Drive to %s" "$indexer_path"
    download_from_gdrive "$indexer_download_id" "$indexer_path"
fi
if [ ! -e "$model_path" ]; then
    printf "\n\nDownloading model checkpoint from Google Drive to %s" "$model_path"
    download_from_gdrive "$model_download_id" "$model_path"
fi

echo
python -m aida_utexas.neural.index \
    "$working_dir" --indexer_path "$indexer_path" \
    $optional_args

echo
python -m aida_utexas.neural.gen_hypoth \
    "$working_dir" --indexer_path "$indexer_path" --model_path "$model_path" --device="$device" \
    $optional_args
