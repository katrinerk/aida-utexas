# The UTexas system for the DARPA AIDA project (TA3)

## Data Preparation

* Prepare an input directory <input_dir> with TA2 KBs, each in a sub-directory named by the TA2 run name. For example:
```
<input_dir>
├── GAIA_TA1.Colorado_TA2
│   └── NIST
│       └── GAIA_TA1.Colorado_TA2.ttl
└── OPERA_TA1.OPERA_TA2
    └── NIST
        └── OPERA_TA1.OPERA_TA2.ttl
```
* Prepare a SIN directory <sin_dir> with Statement of Information Need (SIN) xml files. For example:
```
<sin_dir>
├── T201.xml
├── T202.xml
└── T203.xml
```

## Local Experiments

### Dependencies

* (Recommended) Create a virtual environment with either virtualenv or Conda.
* Install PyTorch: https://pytorch.org/get-started/locally/.
* Install other python dependencies by `pip install -r requirements.txt`.

### Download the Latest Model Checkpoints

```
./scripts/download_models.sh --force
```

### Run Pipeline

```
INPUT=<input_dir> SIN=<sin_dir> OUTPUT=<output_dir> \
    ./scripts/run_simple.sh <TA2_run_name> <TA3_run_name> [optional_args]
```

* `<input_dir>` and `<sin_dir>` are as described in the [Date Preparation](#data-preparation) section.
* `<output_dir>` is an output directory to write intermediate results and final hypotheses.
* `<TA2_run_name>` is the full TA2 run name, for example, `GAIA_TA1.Colorado_TA2`.
* `<TA3_run_name>` is the TA3 run name to be appended, for example, `UTexas_1`.
* `[optional_args]` include:
  * `--num_hyps <NUM_HYPS>`: number of hypotheses to produce for each SIN, default = 50
  * `--max_num_hops <MAX_NUM_HOPS>`: maximum number of hops to expand from a cluster seed to extract a subgraph, default = 5
  * `--min_num_eres <MIN_NUM_ERES>`: minimum number of EREs to  stop subgraph expansion, default = 100
  * `--min_num_stmts <MIN_NUM_STMTS>`: minimum number of statements to stop subgraph expansion, default = 200
  * `--coref_compress`: when specified, first compress ERE coreference on the input TA2 KB
  * `--plaus_rerank`: when specified, use a plausibility classifier to rerank hypothesis seeds
  * `--device`: which CUDA device to use for the neural module, default = -1 (CPU)
  * `--sin_id_prefix <SIN_ID_PREFIX>`: the prefix of SIN IDs to use in naming the final hypotheses, default = AIDA_M36_TA3
  * `--force`: if specified, overwrite existing output files without warning


After execution, the final hypotheses would be written to `<output_dir>/<TA2_run_name>.<TA3_run_name>/`, for example, `<output_dir>/GAIA_TA1.Colorado_TA2.UTexas_1/`. The intermediate files would be written to `<output_dir>/WORKING/<TA2_run_name>.<TA3_run_name>/`.
