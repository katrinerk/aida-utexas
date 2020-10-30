#!/bin/bash

force=false

if [ "$1" == "--force" ]; then
    force=true
fi

echo
echo

indexer_url="https://drive.google.com/file/d/1iwHEiaz_uUbHObFETYYTLLkXxW0_SjlC/view?usp=sharing"
indexer_path="resources/indexers.p"

gcn_model_url="https://drive.google.com/file/d/1oC2rZnq6Sgnn2hsj4fC0eVuQCUCIRAuk/view?usp=sharing"
gcn_model_path="resources/gcn2-cuda_best_5000_1.ckpt"

plaus_model_url="https://drive.google.com/file/d/1g8_SUtHf--vV25XEtReB_xEU-Eilmb5B/view?usp=sharing"
plaus_model_path="resources/plaus_check.ckpt"

gdown_url="https://raw.githubusercontent.com/circulosmeos/gdown.pl/master/gdown.pl"

if $force || [ ! -e "$indexer_path" ]; then
    printf "\n\nDownloading indexer file from Google Drive to %s" "$indexer_path"
    perl <(curl -fSsL "$gdown_url") "$indexer_url" "$indexer_path"
fi
if $force || [ ! -e "$gcn_model_path" ]; then
    printf "\n\nDownloading GCN model checkpoint from Google Drive to %s" "$gcn_model_path"
    perl <(curl -fSsL "$gdown_url") "$gcn_model_url" "$gcn_model_path"
fi
if $force || [ ! -e "$plaus_model_path" ]; then
    printf "\n\nDownloading plausibility model checkpoint from Google Drive to %s" "$plaus_model_path"
    perl <(curl -fSsL "$gdown_url") "$plaus_model_url" "$plaus_model_path"
fi
