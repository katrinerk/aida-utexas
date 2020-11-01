import itertools
import json
import logging
import math
from argparse import ArgumentParser
from collections import Counter, defaultdict
from typing import Dict, List, Set

from aida_utexas import util
from aida_utexas.aif import JsonGraph
from aida_utexas.hypothesis import HypothesisSeed, AidaHypothesisCollection
from pipeline.training.graph_salads.plaus_classifier import eval_plaus


def rerank_seeds_by_plausibility(seeds_by_facet: Dict, graph_path: str,
                                 plausibility_model_path: str = None, indexer_path: str = None):
    logging.info('Re-rank hypothesis seeds by plausibility ...')

    all_seed_stmts_list = [s.hypothesis.stmts for seeds in seeds_by_facet.values() for s in seeds]

    all_scores = eval_plaus(
        indexer_info_file=indexer_path,
        model_path=plausibility_model_path,
        kb_path=graph_path,
        list_of_clusters=all_seed_stmts_list
    )

    offset = 0

    for facet_label, seeds in seeds_by_facet.items():
        for seed, score in zip(seeds, all_scores[offset:]):
            seed.plausibility_score = score
        offset += len(seeds)

    return {facet_label: sorted(seeds, key=lambda s: s.get_scores(), reverse=True)
            for facet_label, seeds in seeds_by_facet.items()}


def select_seeds_by_novelty(seeds_by_facet: Dict, max_num_seeds: int) -> List[HypothesisSeed]:
    logging.info(f'Select top {max_num_seeds} seeds across all facets with maximum novelty ...')

    logging.info('Ranking facets by their top hypothesis seeds ...')
    seeds_by_facet = {facet: seeds for facet, seeds in seeds_by_facet.items() if len(seeds) > 0}
    seeds_by_facet = dict(sorted(seeds_by_facet.items(), key=lambda pair: pair[1][0].get_scores()))

    # the characterization of query variable fillers for already selected seeds.
    # format: a mapping from each qvar to a counter of fillers
    # we penalize a seed that has the same qvar filler that we have seen before, with a value
    # equivalent to the number of previous seeds that had the same filler.
    qvar_characterization = defaultdict(Counter)

    # update qvar_characterization by counting the qvar fillers of a new hypothesis seed
    def _update_qvar_characterization(_seed: HypothesisSeed):
        for _qvar, _filler in _seed.qvar_filler.items():
            # do not include entry points: novelty in entry points is not rewarded
            if _qvar not in _seed.entrypoints:
                qvar_characterization[_qvar][_filler] += 1

    selected_seeds = []

    available_indices_by_facet = {
        facet_label: set(range(len(seeds))) for facet_label, seeds in seeds_by_facet.items()}

    done_facets = set()

    # we cycle through the different facets, and pick at most one seed from each facet at a time,
    # with the assumption that seeds in different facets should always exhibit higher novelty
    # than seeds in the same facet
    for facet_label in itertools.cycle(seeds_by_facet.keys()):
        # if all facets are marked as done (no more new seeds), break the loop
        if len(done_facets) == len(seeds_by_facet):
            break

        # if no seed has been selected, simply go with the seed with highest score in this facet
        if len(selected_seeds) == 0:
            next_seed_idx = 0 if len(seeds_by_facet[facet_label]) > 0 else None
        else:
            # select the next best seed from this facet
            next_seed_idx = choose_next_novel_seed(
                seeds=seeds_by_facet[facet_label],
                available_indices=available_indices_by_facet[facet_label],
                qvar_characterization=qvar_characterization)

        # if there is no new seed in this facet, mark this facet as done and skip to the next one
        if next_seed_idx is None:
            done_facets.add(facet_label)
            continue

        # append the next best item to selected_seeds
        next_seed = seeds_by_facet[facet_label][next_seed_idx]
        selected_seeds.append(next_seed)

        # update qvar_characterization
        _update_qvar_characterization(next_seed)

        # remove the selected index from available_indices
        available_indices_by_facet[facet_label].discard(next_seed_idx)

        # if we have already selected max_num_seeds seeds, break the loop
        if len(selected_seeds) >= max_num_seeds:
            break

    return selected_seeds


def choose_next_novel_seed(seeds: List[HypothesisSeed], available_indices: Set,
                           qvar_characterization: Dict, bonus_for_novel_filler: int = 5):
    best_index, best_score = None, float('-inf')

    # for each seed, determine its novelty score by comparing to qvar_characterization
    for index, seed in enumerate(seeds):
        # if the current index is already selected, skip it
        if index not in available_indices:
            continue

        score = 0

        for qvar, filler in seed.qvar_filler.items():
            # do not count entry point variables when checking for novelty
            if qvar in seed.entrypoints:
                continue

            filler_count = qvar_characterization[qvar][filler]

            # if some higher-ranked seeds have the same filler for this qvar, take a penalty
            if filler_count > 0:
                score -= filler_count
            # otherwise, take a bonus for the novel qvar filler
            else:
                score += bonus_for_novel_filler

        # at this point we have the score for the current seed.
        # if it is the maximally achievable score, stop here and go with this seed
        if score >= bonus_for_novel_filler * len(qvar_characterization):
            best_index = index
            break

        # if the score is better than the previous best, record this index as the best one
        if score > best_score:
            best_index = index
            best_score = score

    return best_index


def main():
    parser = ArgumentParser()
    parser.add_argument('graph_path', help='Path to the input graph JSON file')
    parser.add_argument('raw_seeds_path',
                        help='Path to the raw hypothesis seeds file, or a directory with '
                             'multiple seeds files')
    parser.add_argument('output_dir', help='Directory to write the reranked hypothesis seeds')
    parser.add_argument('--plausibility_model_path', help='Path to a hypothesis plausibility model')
    parser.add_argument('--indexer_path', help="Path to the indexers file")
    parser.add_argument('-n', '--max_num_seeds', type=int, default=None,
                        help='Only output up to n hypothesis seeds')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='If specified, overwrite existing output files without warning')

    args = parser.parse_args()

    json_graph = JsonGraph.from_dict(util.read_json_file(args.graph_path, 'JSON graph'))

    raw_seeds_file_paths = util.get_file_list(args.raw_seeds_path, suffix='.json', sort=True)

    output_dir = util.get_output_dir(args.output_dir, overwrite_warning=not args.force)

    for raw_seeds_file_path in raw_seeds_file_paths:
        raw_seeds_json = util.read_json_file(raw_seeds_file_path, 'seeds by facet')
        seeds_by_facet = {}
        for facet_label, seeds_json in raw_seeds_json.items():
            if facet_label != 'graph':
                seeds_by_facet[facet_label] = [HypothesisSeed.from_json(seed_json, json_graph)
                                               for seed_json in seeds_json]

        if args.plausibility_model_path is not None and args.indexer_path is not None:
            seeds_by_facet = rerank_seeds_by_plausibility(
                seeds_by_facet, args.graph_path, args.plausibility_model_path, args.indexer_path)

        seeds = select_seeds_by_novelty(seeds_by_facet, args.max_num_seeds)

        hypotheses_to_export = []

        # turn ranks into the log weights of seed hypotheses
        # meaningless numbers. just assign 1/2, 1/3, 1/4, ...
        for rank, seed in enumerate(seeds):
            seed.hypothesis.update_weight(math.log(1.0 / (rank + 1)))
            hypotheses_to_export.append(seed.finalize())

        hypothesis_collection = AidaHypothesisCollection(hypotheses_to_export)

        seeds_json = hypothesis_collection.to_json()
        seeds_json['graph'] = raw_seeds_json['graph']

        output_path = output_dir / (raw_seeds_file_path.name.split('_')[0] + '_seeds.json')
        logging.info('Writing re-ranked hypothesis seeds to {} ...'.format(output_path))
        with open(str(output_path), 'w') as fout:
            json.dump(seeds_json, fout, indent=1)


if __name__ == '__main__':
    main()
