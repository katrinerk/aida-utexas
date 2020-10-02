# The UTexas system for the DARPA AIDA project (TA3)

## Data Preparation

* Prepare an input directory <input_dir> with TA2 KBs, each in a sub-directory named by the TA2 run name. For example:
```
<input_dir>
├── OPERA_1.Colorado_2
│   └── NIST
│       └── OPERA_1.Colorado_2.ttl
└── LDC_2.LDC_2
    └── NIST
        └── LDC_2.LDC_2.ttl
```
* Prepare a SIN directory <sin_dir> with Statement of Information Need (SIN) xml files. For example:
```
<sin_dir>
├── E101.xml
├── E102.xml
└── E103.xml
```

## Local Experiments

### Dependencies

* Install the latest Apache Jena package: https://jena.apache.org/download/index.cgi.
* Set environment variables for command line usage: https://jena.apache.org/documentation/tools/.
* Install python dependencies by `pip install -r requirements.txt` (virtualenv or Conda environment recommended).

### Run Model

```
INPUT=<input_dir> SIN=<sin_dir> OUTPUT=<output_dir> \
    ./scripts/run_all.sh <TA2_run_name> <TA3_run_name> [optional_args]
```

* `<input_dir>` and `<sin_dir>` are as described in the [Date Preparation](#data-preparation) section.
* `<output_dir>` is an output directory to write intermediate results and final hypotheses.
* `<TA2_run_name>` is the full TA2 run name, for example, `OPERA_1.Colorado_2`.
* `<TA3_run_name>` is the TA3 run name to be appended, for example, `UTexas_1`.
* `[optional_args]` include:
  * `--num_hyps <NUM_HYPS>`: number of hypotheses to produce for each SIN, default = 50
  * `--num_hops <NUM_HOPS>`: number of hops to expand from a cluster seed to extract a subgraph, default = 2
  * `--coref_compress`: when specified, first compress ERE coreference on the input TA2 KB
  * `--device`: which CUDA device to use for the neural module, default = -1 (CPU)
  * `--sin_id_prefix <SIN_ID_PREFIX>`: the prefix of SIN IDs to use in naming the final hypotheses, default = AIDA_M36_TA3
  * `--force`: if specified, overwrite existing output files without warning


After execution, the final hypotheses would be written to `<output_dir>/<TA2_run_name>.<TA3_run_name>/`, for example, `<output_dir>/OPERA_1.Colorado_2.UTexas_1/`. The intermediate files would be written to `<output_dir>/WORKING/<TA2_run_name>.<TA3_run_name>/`.

### Docker

* The latest version of docker image can be found at: https://hub.docker.com/repository/docker/pxch/aida-utexas.
* With the docker image, you can run the system without configuring dependencies:
```
docker run \
    -v <input_dir>:/input:ro \
    -v <sin_dir>:/sin:ro \
    -v <output_dir>:/output \
    -e INPUT=/input \
    -e SIN=/sin \
    -e OUTPUT=/output \
    pxch/aida-utexas:0.3.0 \
    bash /aida-utexas/scripts/run_all.sh <TA2_run_name> <TA3_run_name> [optional_args]
```
* See the [Run Model](#run-model) section for definitions of `<input_dir>`, `<sin_dir>`, `<output_dir>`, `<TA2_run_name>`, `<TA3_run_name>`, and `[optional_args]`.
