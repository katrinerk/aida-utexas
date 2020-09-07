import sys
from pathlib import Path

from aida_utexas import util

seeds_dir = Path(sys.argv[1]).resolve()
assert seeds_dir.exists() and seeds_dir.is_dir(), \
    '{} does not exist!'.format(seeds_dir)

seeds_file_list = sorted(
    [f for f in seeds_dir.iterdir() if f.suffix == '.json'])

for seeds_file in seeds_file_list:
    seeds_json = util.read_json_file(seeds_file, 'seeds')
    num_failed_queries_list = []
    for cluster_idx, cluster in enumerate(seeds_json['support']):
        num_failed_queries = len(cluster['failedQueries'])
        num_failed_queries_list.append(num_failed_queries)

        num_query_stmts = len(cluster['queryStatements'])
        num_stmts = len(cluster['statements'])
        num_stmts_zero_weight = len(
            [stmt_weight for stmt_weight in cluster['statementWeights']
             if stmt_weight == 0])
        num_stmts_negative_weight = len(
            [stmt_weight for stmt_weight in cluster['statementWeights']
             if stmt_weight == -100])
        print(
            'Cluster #{}: # query stmts = {}, # stmts = {}, # stmt (0) = {}, '
            '# stmts (-100) = {}'.format(
                cluster_idx, num_query_stmts, num_stmts, num_stmts_zero_weight,
                num_stmts_negative_weight))

    print('Across all {} hypothesis seeds'.format(len(num_failed_queries_list)))
    print('Number of failed queries: min = {}, max = {}, average = {:.2f}'.format(
        min(num_failed_queries_list),
        max(num_failed_queries_list),
        sum(num_failed_queries_list) / len(num_failed_queries_list)))
