# The UTexas system for the DARPA AIDA project (TA3 Evaluation for 2022)

## Data Preparation

* Prepare an aif directory <aif_dir> with one TA2 KB and an query directory <query_dir> with sub-directories containing queries and topics for different conditions. For example:
```
<input_dir>
├── XXX.ttl
└── Queries
    ├── Condition5
    │     ├── Query_Claim_Frames
    │     │   ├── CLL0C04G510.000030.ttl
    │     │   └── ...
    │     └── topics.tsv
    ├── Condition6
    └── Condition7
    
```

## Local Experiments

### Dependencies

* (Recommended) Create a virtual environment with either virtualenv or Conda.
* Install PyTorch: https://pytorch.org/get-started/locally/.
* Install other python dependencies by `pip install -r requirements.txt`.
* Download NLI model checkpoint by 
```
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
tar -xzvf roberta.large.tar.gz
```


### Run Pipeline

```
./scripts/run_all_22.sh <AIF> <AIFNAME> <WORKSPACE> <RUN> <CONDITION> [optional_args]
```

    printf "* <AIF>: the path to original AIF data\n"
    printf "* <AIFNAME>: the file of name of original AIF data, the format of the name should be 'XXXXX.ttl'\n" #add new input
    printf "* <WORKSPACE>: the path to the working directory to do main process\n"
    printf "* <RUN>: the name of our run, i.e., ta2_colorado\n"
    printf "* <CONDITION>: the condition to run on , i.e., condition5, condition6\n"

    printf "* --threshold THRESHOLD: the score threshold to determine relatedness/independence, default = 0.58\n"
    printf "* --query QUERY: the path to original query data, default = None\n"

* `<AIF>` is the path to original AIF data, as the <aif_dir> described above.
* `<AIFNAME>` is the file of name of original AIF data, the format of the name should be 'XXXXX.ttl'".
* `<WORKSPACE>` is the path to the working directory to do main process.
* `<RUN>` is the name of our run, i.e., `ta2_colorado`.
* `<CONDITION>` is the condition to run on , i.e., `condition5`, `condition6`
* `[optional_args]` include:
  * `--threshold <THRESHOLD>`: the score threshold to determine relatedness/independence, default = 0.58
  * `--query <QUERY>`: the path to original query data, default = None, as the <query_dir> described above.


While execution, the intermediate results would be written to pre-defined directory structure. 

```
<WORKSPACE>
└── <RUN>
    └── <CONDITION>
          ├── step1_query_claim_relatedness
          │   ├── q2d_relatedness.csv
          │   └── ...
          ├── step2_query_claim_nli
          │   ├── claim_claim.csv
          |   ├── d2d_nli.csv / q2d_nli.csv
          │   └── ...
          └── step3_claim_claim_ranking
              ├── claim_claim_relatedness.csv
              └── ...
```

Finally, for each query claim, the pipeline will generate corresponding claims and a ranking file.
```
<WORKSPACE>
└── <out>
    └── <output>
        └── <RUN>
            └── <NIST>
                └── <CONDITION>
                    ├── <Query_Claim.ttl>
                    │   ├── <xxx.ttl>
                    │   └── ...
                    └── xxx.ranking.tsv

An example would be:
<WORKSPACE>
└── <out>
    └── <output>
        └── <ta2_gaia_high_recall>
            └── <NIST>
                └── <Condition5>
                    ├── <CLL0C04C95X.000004>
                    │   ├── <claim_L0C04CAHC_1.ttl>
                    │   └── ...
                    └── CLL0C04C95X.000004.ranking.tsv
```
