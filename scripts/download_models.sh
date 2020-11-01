#!/bin/bash

force=false

if [ "$1" == "--force" ]; then
    force=true
fi

echo
echo

python -m pip install gdown

indexer_url="https://drive.google.com/uc?id=1iwHEiaz_uUbHObFETYYTLLkXxW0_SjlC"
#indexer_url="https://drive.google.com/file/d/1iwHEiaz_uUbHObFETYYTLLkXxW0_SjlC/view?usp=sharing"
indexer_path="resources/indexers.p"

gcn_model_url="https://drive.google.com/uc?id=1oC2rZnq6Sgnn2hsj4fC0eVuQCUCIRAuk"
#gcn_model_url="https://drive.google.com/file/d/1oC2rZnq6Sgnn2hsj4fC0eVuQCUCIRAuk/view?usp=sharing"
gcn_model_path="resources/gcn2-cuda_best_5000_1.ckpt"

plaus_model_url="https://drive.google.com/uc?id=1g8_SUtHf--vV25XEtReB_xEU-Eilmb5B"
#plaus_model_url="https://drive.google.com/file/d/1g8_SUtHf--vV25XEtReB_xEU-Eilmb5B/view?usp=sharing"
plaus_model_path="resources/plaus_check.ckpt"

#gdown_url="https://raw.githubusercontent.com/circulosmeos/gdown.pl/master/gdown.pl"

if $force || [ ! -e "$indexer_path" ]; then
    printf "\n\nDownloading indexer file from Google Drive to %s" "$indexer_path"
    gdown "$indexer_url" -O "$indexer_path"
fi
if $force || [ ! -e "$gcn_model_path" ]; then
    printf "\n\nDownloading GCN model checkpoint from Google Drive to %s" "$gcn_model_path"
    gdown "$gcn_model_url" -O "$gcn_model_path"
fi
if $force || [ ! -e "$plaus_model_path" ]; then
    printf "\n\nDownloading plausibility model checkpoint from Google Drive to %s" "$plaus_model_path"
    gdown "$plaus_model_url" -O "$plaus_model_path"
fi
