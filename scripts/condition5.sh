#!/bin/bash


WORKSPACE=$1
RUN=$2
THRESHOLD=$3
CONDITION=$4


## query-claim relatedness
echo "Start query-claim relatedness calculation..."

python -m aida_utexas.redundancy.redundancy_score \
--run ${RUN} --type query_claim \
--condition ${CONDITION} --threshold ${THRESHOLD} \
--docclaim_dir ${WORKSPACE}/preprocessed \
--query_dir ${WORKSPACE}/preprocessed/query \
--workspace ${WORKSPACE}/working

echo "Finish"



## pair match
echo "Start finding topic matched query-claim pair..."

python -m aida_utexas.redundancy.pair_matcher \
--run ${RUN} --condition ${CONDITION} \
--docclaim_dir ${WORKSPACE}/preprocessed \
--query_dir ${WORKSPACE}/preprocessed/query \
--workspace ${WORKSPACE}/working

echo "Finish"



## NLI on matched query-claim pair
echo "Start NLI on matched query-claim pair..."

python -m aida_utexas.NLI.nli_processor \
--run ${RUN} --type query_claim \
--condition ${CONDITION} \
--workspace ${WORKSPACE}/working

echo "Finish"


## query_related_claims.csv and claim_claim.csv
echo "Start writing query_related_claims.csv and claim_claim.csv..."

python -m aida_utexas.redundancy.query_related_claims \
--query_dir ${WORKSPACE}/preprocessed/query \
--docclaim_dir ${WORKSPACE}/preprocessed \
--working_dir ${WORKSPACE}/working \
--run_id ${RUN} --condition ${CONDITION} \
--threshold ${THRESHOLD} -f

echo "Finish"


## claim-claim relatedness
echo "Start claim-claim relatedness calculation..."

python -m aida_utexas.redundancy.redundancy_score \
--run ${RUN} --type claim_claim \
--condition ${CONDITION} --threshold ${THRESHOLD} \
--docclaim_dir ${WORKSPACE}/preprocessed \
--query_dir ${WORKSPACE}/preprocessed/query \
--workspace ${WORKSPACE}/working

echo "Finish"



