#!/bin/bash

set -e

print_usage() {
    printf "Usage: run_all.sh INPUT_RUN_NAME RUN_NAME "
    printf "[--num_hyps NUM_HYPS] [--num_hops NUM_HOPS] [--coref_compress] [--device DEVICE] [--sin_id_prefix PREFIX] [--force]\n"
    printf "* <INPUT_RUN_NAME>: the RUN_NAME of a TA2 KB as the input, i.e., LDC_2.LDC_2\n"
    printf "* <RUN_NAME>: the name of our run, which will be appended to <input_run_name> to get the output RUN_NAME, i.e., UTexas_1\n"
    printf "* --num_hyps <NUM_HYPS>: number of hypotheses to produce for each SIN, default = 14\n"
    printf "* --num_hops <NUM_HOPS>: number of hops to expand from a cluster seed to extract a subgraph, default = 2\n"
    printf "* --coref_compress: when specified, first compress ERE coreference on the input TA2 KB\n"
    printf "* --device: which CUDA device to use for the neural module, default = -1 (CPU)\n"
    printf "* --sin_id_prefix: the prefix of SIN IDs to use in naming the final hypotheses, default = AIDA_M18_TA3\n"
    printf "* --force: if specified, overwrite existing output files without warning\n"
}

parse_args() {
    if [ $# -lt 2 ]; then
        print_usage
        exit 1
    fi

    input_run_name=$1
    run_name=$2
    shift
    shift

    num_hyps=14
    num_hops=2
    do_coref_compression=false
    device=-1
    sin_id_prefix="AIDA_M18_TA3"
    force_overwrite=false

    while [ "$1" != "" ]; do
        case $1 in
        --num_hyps)
            shift
            num_hyps=$1
            ;;
        --num_hops)
            shift
            num_hops=$1
            ;;
        --coref_compress)
            do_coref_compression=true
            ;;
        --device)
            shift
            device=$1
            ;;
        --sin_id_prefix)
            shift
            sin_id_prefix=$1
            ;;
        --force)
            force_overwrite=true
            ;;
        *)
            print_usage
            exit 1
            ;;
        esac
        shift
    done

    input_kb_path=$INPUT/$input_run_name/NIST/$input_run_name.ttl
    output_dir=$OUTPUT/$input_run_name.$run_name
    working_dir=$OUTPUT/WORKING/$input_run_name.$run_name
}

failfast() {
    if [ "$INPUT" == "" ]; then
        printf "\$INPUT variable not set, exit ...\n"
        exit 1
    fi
    if [ "$SIN" == "" ]; then
        printf "\$SIN variable not set, exit ...\n"
        exit 1
    fi
    if [ "$OUTPUT" == "" ]; then
        printf "\$OUTPUT variable not set, exit ...\n"
        exit 1
    fi
    if [ ! -e "$input_kb_path" ]; then
        printf "Cannot find %s, exit ...\n" "$input_kb_path"
        exit 1
    fi
}

print_args() {
    printf "\n\$INPUT set to: %s\n" "$INPUT"
    printf "INPUT_RUN_NAME: %s\n" "$input_run_name"
    printf "\tUsing TA2 KB from: %s\n" "$input_kb_path"

    printf "\n\$SIN set to: %s\n" "$SIN"
    printf "\tFound SIN files:\n"
    for f in "$SIN"/*.xml; do
        printf "\t\t%s\n" "$f"
    done

    printf "\n\$OUTPUT set to: %s\n" "$OUTPUT"
    printf "RUN_NAME: %s\n" "$run_name"
    printf "\tMaking output directory: %s\n" "$output_dir"
    mkdir -p "$output_dir"
    printf "\tMaking working directory: %s\n" "$working_dir"
    mkdir -p "$working_dir"

    printf "\nOptional parameters:\n"
    printf "+ Number of hypotheses per SIN: %s\n" "$num_hyps"
    printf "+ Number of hops to extract subgraphs: %s\n" "$num_hops"
    printf "+ Do coref compression?: %s\n" "$do_coref_compression"
    printf "+ Device for neural module: %s\n" "$device"
    printf "+ Prefix of SIN IDs: %s\n" "$sin_id_prefix"
    printf "+ Force overwrite?: %s\n" "$force_overwrite"
}

parse_args "$@"
failfast
print_args

optional_args=()
$force_overwrite && optional_args+=( "--force" )
optional_args="${optional_args[@]}"

echo
printf "+ Optional arguments for python scripts: %s\n" "$optional_args"

echo
echo
echo Start preprocessing.sh ...

./scripts/preprocessing.sh \
    "$input_run_name" "$input_kb_path" "$working_dir" "$num_hyps" "$num_hops" \
    "$do_coref_compression" "$optional_args"

echo
echo
echo Start neural.sh ...

./scripts/neural.sh "$working_dir" "$device" "$optional_args"

echo
echo
echo Start postprocessing.sh ...

./scripts/postprocessing.sh \
    "$input_run_name" "$run_name" "$output_dir" "$working_dir" "$num_hyps" \
    "$do_coref_compression" "$sin_id_prefix" "$optional_args"
