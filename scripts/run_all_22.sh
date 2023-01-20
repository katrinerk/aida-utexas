#!/bin/bash

print_usage() {
    printf "Usage: run_all_22.sh AIF AIFNAME WORKSPACE RUN CONDITION"
    printf "[--threshold THRESHOLD] [--query QUERY]"
    printf "* <AIF>: the path to original AIF data\n"
    printf "* <AIFNAME>: the file of name of original AIF data, the format of the name should be 'XXXXX.ttl'\n" #add new input
    #printf "* <QUERY>: the path to original query data, if you don't have query, type None\n"
    printf "* <WORKSPACE>: the path to the working directory to do main process\n"
    printf "* <RUN>: the name of our run, i.e., ta2_colorado\n"
    printf "* <CONDITION>: the condition to run on , i.e., condition5, condition6\n"

    printf "* --threshold THRESHOLD: the score threshold to determine relatedness/independence, default = 0.58\n"
    printf "* --query QUERY: the path to original query data, default = None\n"

}


parse_args() {
    if [ $# -lt 5 ]; then
        print_usage
        exit 1
    fi

    AIF=$1
    AIFNAME=$2
    WORKSPACE=$3
    RUN=$4
    CONDITION=$5
    shift
    shift
    shift
    shift
    shift

    THRESHOLD=0.58
    QUERY="None"

    while [ "$1" != "" ]; do
        echo "$1"
        case "$1" in
        --threshold)
            THRESHOLD=$2
            shift
            shift
            ;;
        --query)
            QUERY=$2
            shift
            shift
            echo "get query input"
            ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
        *)
            ;;
        esac
    done
}

sanity_check(){
    if [ "$CONDITION" != "condition5" ] && [ "$CONDITION" != "condition6" ] && [ "$CONDITION" != "condition7" ]; then
        printf "CONDITION must be condition5, condition6 or condition7"
        exit 1
    fi
}


print_args() {
    printf "AIF: %s\n" "$AIF"
    printf "AIF name: %s\n" "$AIFNAME" #aifname input display
    printf "QUERY: %s\n" "$QUERY"
    printf "* <QUERY>: the path to original query data\n"
    printf "WORKSPACE: %s\n" "$WORKSPACE"
    printf "\tCreating workspace... \n"
    mkdir -p "$WORKSPACE"
    printf "RUN: %s\n" "$RUN"
    printf "CONDITION: %s\n" "$CONDITION"
    printf "threshold to classify related/independent claims: %f\n" "$THRESHOLD"
}


parse_args "$@"
sanity_check
print_args


parentdir="$(dirname "$PWD")"
export PYTHONPATH="${PYTHONPATH}:$parentdir"

#echo ${PYTHONPATH}

## preprocess
echo "Start preprocessing..."
pre_out_path="$WORKSPACE/preprocessed/$RUN"
query_out_path="$WORKSPACE/preprocessed/query"

cd pipeline/preprocessing

if [ "$QUERY" == "None" ]; then 
    python3 -m preprocess_claims_relativepath --aif_path $AIF --doc_output_dir $pre_out_path
else
    python3 -m preprocess_claims_relativepath --aif_path $AIF --doc_output_dir $pre_out_path -q $QUERY -Q $query_out_path
fi

echo "Finish"

cd ../..

if [ "$CONDITION" = "condition5" ]; then
    echo "Start condition5.sh ..."
    ./scripts/condition5.sh \
        "$WORKSPACE" "$RUN" "$THRESHOLD" "$CONDITION"
    echo "Finish"

elif [ "$CONDITION" = "condition6" ]; then
    echo "Start condition6.sh ..."
    ./scripts/condition6.sh \
        "$WORKSPACE" "$RUN" "$THRESHOLD" "$CONDITION"
    echo "Finish"

elif [ "$CONDITION" = "condition7" ]; then
    echo "Start condition7.sh ..."
    ./scripts/condition7.sh \
        "$WORKSPACE" "$RUN" "$THRESHOLD" "$CONDITION"
    echo "Finish"
fi

## postprocess
echo "Start postprocessing..."
output_dir="$WORKSPACE/out"
working_dir="$WORKSPACE/working"
kb_path="$AIF/$AIFNAME"
graph_path="$WORKSPACE/preprocessed/$RUN/json/$AIFNAME.json"

cd pipeline/postprocessing

if [ "$CONDITION" = "condition5" ]; then
    python3 -m produce_claim_aif_c5_addedge_newest $working_dir $RUN condition5 $graph_path $kb_path $output_dir
else 
    if [ "$CONDITION" = "condition6" ]; then 
        python3 -m produce_claim_aif_c6_addedge_newest $working_dir $RUN condition6 $graph_path $kb_path $output_dir
    else 
        python3 -m produce_claim_aif_c6_addedge_newest $working_dir $RUN condition7 $graph_path $kb_path $output_dir
    fi
fi
    echo "Finish"

cd ../..

echo "ALL DONE"