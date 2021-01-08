"""
Author: Katrin Erk, Mar 2019
- Rule-based creation of initial hypotheses.
- This only adds the statements that the Statement of Information Need asks for, but constructs all
possible hypothesis seeds that can be made using different statements that all fill the same SOIN.

Update: Pengxiang Cheng, Aug 2020
- Clean-up and refactoring
- Naming conventions:
    - qvar (query variables): variables in core constraints (edges of a frame)
    - qvar_filler: a dictionary from each qvar to a filler ERE label
    - ep (entry points): variables grounded by entry point descriptors
    - ep_fillers: a list of possible ERE labels to fill an ep variable, from SoIN matching

Update: Pengxiang Cheng, Sep 2020
- Split cluster_seeds.py into two separate files, and rename the classes
- Change to making seeds per facet (an event or relation variable name in a frame)
- Remove early-cutoff and complex entry point combination logic related to it
- Now outputs a dictionary of sorted seeds per facet, which will be reranked based on
    plausibility (optional) and novelty in another script.
"""

import itertools
import logging
from collections import Counter, deque, defaultdict
from operator import itemgetter
from typing import Dict, List, Tuple

from aida_utexas.aif import JsonGraph
from aida_utexas.hypothesis.aida_hypothesis import AidaHypothesis
from aida_utexas.hypothesis.date_check import AidaIncompleteDate
from aida_utexas.hypothesis.hypothesis_seed import HypothesisSeed
from aida_utexas.hypothesis.seed_expansion import HypothesisSeedExpansion


# class that manages all hypothesis seeds
class HypothesisSeedManager:
    # initialize with a JsonGraph object and json representation of a query
    def __init__(self, json_graph: JsonGraph, query_json: dict, frame_grouping: bool = False,
                 discard_failed_core_constraints: bool = False, rank_cutoff: int = None):
        self.json_graph = json_graph
        self.query_json = query_json

        # convert temporal constraints into AidaIncompleteDate
        self.temporal_constraints = self._pythonize_datetime(self.query_json.get('temporal', {}))

        # whether to group query constraints by frames, or by facets (default = False)
        self.frame_grouping = frame_grouping

        # discard hypothesis seeds with any failed core constraints?
        self.discard_failed_core_constraints = discard_failed_core_constraints

        # cut off partially formed hypotheses during creation
        # if there are at least rank_cutoff other hypothesis seeds with the same fillers
        # for rank_cutoff_qvar_count query variables?
        self.rank_cutoff = rank_cutoff
        self.rank_cutoff_qvar_count = 3

    # split the core constraints of the query into a list of facets.
    # note that some of the facets might contain no entry point variables, in such case, we merge
    # them into the closes facets with entry point variables.
    # Katrin December 2020:
    # if a facet ends up containing a variable that is further constrained in
    # another facet, merge the two facets
    def split_facets(self):
        # if grouping is supposed to be by frames (that is, more coarse-grained)
        # then make this grouping
        if self.frame_grouping:
            frame_to_constraints = defaultdict(list)
            for frame in self.query_json['frames']:
                frame_id = frame['frame_id']
                for constraint in frame['edges']:
                    frame_to_constraints[frame_id].append(constraint)
            return frame_to_constraints

        # grouping is more fine-grained, one event at a time
        # all entry point variables in the SoIN
        all_ep_vars = set(self.query_json['ep_matches_dict'].keys())

        # a mapping from each facet label to a list of constraints with the facet as subject
        facet_to_constraints = defaultdict(list)
        # a mapping from each facet label to a set of variables occurred in the facet
        facet_to_variables = defaultdict(set)
        # a mapping from each facet label to subject variables (typically, events and relations)
        facet_to_eventrel = defaultdict(set)
        # all subject variables
        all_eventrel = set()

        for frame in self.query_json['frames']:
            for constraint in frame['edges']:
                facet_label = constraint[1]
                facet_to_constraints[facet_label].append(constraint)
                facet_to_variables[facet_label].add(facet_label)
                facet_to_variables[facet_label].add(constraint[3])
                all_eventrel.add(constraint[1])

        # determine the facets that do / do not contain entry point variables
        facets_with_ep, facets_without_ep = [], []
        for facet_label, variables in facet_to_variables.items():
            if any(var in all_ep_vars for var in variables):
                facets_with_ep.append(facet_label)
            else:
                facets_without_ep.append(facet_label)

        # merge each facet without EP variables to the first facet with EP variables that
        # shares at least one common variable
        for f1 in facets_without_ep:
            for f2 in facets_with_ep:
                if len(facet_to_variables[f2].intersection(facet_to_variables[f1])) > 0:
                    facet_to_constraints[f2] += facet_to_constraints[f1]
                    del facet_to_constraints[f1]
                    break

        # determine facets that contain object variables that are subject variables elsewhere
        facets_depending_on_other_events = [ ]
        for facet_label, variables in facet_to_variables.items():
            if any(var in all_eventrel and var != facet_label for var in variables):
                facets_depending_on_other_events.append(facet_label)

        # merge each facet that has an event as an object
        # to the first face that has this event as a subject
        for f1 in facets_depending_on_other_events:
            for f2, f2_eventvariables in facet_to_eventrel.items():
                if any(var in f2_eventvariables and var not in facet_to_eventrel[f1] for var in variables):
                    facet_to_constraints[f2] += facet_to_constraints[f1]
                    del facet_to_constraints[f1]
                    break

        # Katrin Jan 2021
        # remove duplicate constraints
        # that only differ in their SoIN edge label
        new_facet_to_constraints = defaultdict(list)
        for facetlabel, constraints in facet_to_constraints.items():
            for constraint in constraints:
                # take the constraint apart into its pieces
                edgelabel, subj1, pred1, obj1, objtype1 = constraint

                # check if it is a duplicate
                if any(subj1 == subj2 and pred1 == pred2 and obj1 == obj2 and objtype1 == objtype2 for el2, subj2, pred2, obj2, objtype2 in new_facet_to_constraints[facetlabel]):
                    # duplicate: ditch this constraint
                    pass
                else:
                    # not a duplicate
                    new_facet_to_constraints[facetlabel].append(constraint)

            # testing: did we remove any duplicates?
            # if len(new_facet_to_constraints[facetlabel]) < len(facet_to_constraints[facetlabel]):
            #     logging.info(f'HIER removed duplicate constraints reduced from {len(facet_to_constraints[facetlabel])} to {len(new_facet_to_constraints[facetlabel])}')

        
        return new_facet_to_constraints

    # create initial hypothesis seeds
    def make_seeds(self):
        facet_to_constraints = self.split_facets()

        seeds_by_facet = {}

        seed_expansion_obj = HypothesisSeedExpansion()

        for facet_label, core_constraints in facet_to_constraints.items():

            logging.info(f'Creating hypothesis seeds for facet {facet_label} ...')
            seeds = self._create_seeds(core_constraints)
            
            # testing
            # for fc, subj, pred, obj, objtype in core_constraints:
            #     logging.info(f'HIER0 original cc {fc} {subj} {pred} {obj} {objtype}')
                
            # Katrin December 2020: adding query expansion here.
            # arguments: core constraints, temporal constraints, entry points
            additional_coreconstraint_lists, additional_temporal = seed_expansion_obj.expand(core_constraints, self.temporal_constraints, \
                                                                                             list(self.query_json['ep_matches_dict'].keys()))
            self.temporal_constraints.update(additional_temporal)

            
            query_expansion_count = 0
            for cc in additional_coreconstraint_lists:
                additional = self._create_seeds(cc)
                query_expansion_count += len(additional)
                seeds += additional
            logging.info(f'Query expansion added {query_expansion_count} seeds.')

            seeds_by_facet[facet_label] = sorted(
                seeds, key=lambda s: s.get_scores(), reverse=True)

        return seeds_by_facet

    # create initial hypothesis seeds
    def _create_seeds(self, core_constraints: List[Tuple]) -> List[HypothesisSeed]:
        # the queue of seeds-in-making
        seeds_todo = deque()
        # the list of finished seeds
        seeds_done = []

        ep_matches_dict = self.query_json['ep_matches_dict']

        # entry point variables occurring in the core constraints: each core constraint has the form
        # [subj, pred, obj, obj_type], where only obj can be potentially entry point variables
        ep_var_list = sorted([obj for _, _, _, obj, _ in core_constraints if obj in ep_matches_dict])

        # entry point fillers and weights filtered and reranked by both ep match scores and
        # role match scores
        reranked_ep_matches_dict = {}

        # have we found any hypothesis without failed core constraints yet?
        # if so, we can eliminate all hypotheses with failed core constraints
        found_hypothesis_wo_failed = False

        for ep_var in ep_var_list:
            ep_matches = ep_matches_dict[ep_var]

            ep_scores = {}

            fillers_filtered_both = []
            fillers_filtered_role_score = []

            # Katrin December 2020: possibly reduce the number of entry points we keep
            # (not done yet, but consider doing it)
            for ep_filler, ep_weight in ep_matches:
                ep_role_score = self._entrypoint_filler_role_score(
                    ep_var, ep_filler, core_constraints)
                ep_scores[ep_filler] = (ep_weight, ep_role_score)
                # an ep_role_score > 0 means there is at least one role match for the filler
                if ep_role_score > 0:
                    fillers_filtered_role_score.append(ep_filler)
                    # an ep_weight > 50 is considered as a "good" match (max score = 100)
                    if ep_weight > 50.0:
                        fillers_filtered_both.append(ep_filler)

            # Katrin December 2020: sample down the 20 Caracases
            if len(fillers_filtered_both) > 3:
                logging.info(f'Too many good fillers: sampling down.')
                fillers_filtered_both = fillers_filtered_both[:2]
                
            if len(fillers_filtered_both) > 0:
                logging.info(f'Entry point {ep_var}: kept {len(fillers_filtered_both)} fillers '
                             f'with both SoIN weight > 50 and role score > 0')
                fillers_to_keep = [(f, ep_scores[f][0] * ep_scores[f][1])
                                   for f in fillers_filtered_both]
            elif len(fillers_filtered_role_score) > 0:
                logging.info(f'Entry point {ep_var}: kept {len(fillers_filtered_role_score)} '
                             f'fillers with role score > 0')
                fillers_to_keep = [(f, ep_scores[f][0] * ep_scores[f][1])
                                   for f in fillers_filtered_role_score]
            else:
                logging.info(f'Entry point {ep_var}: kept all {len(ep_matches)} fillers '
                             f'(no filler with role score > 0)')
                fillers_to_keep = ep_matches

            reranked_ep_matches_dict[ep_var] = \
                sorted(fillers_to_keep, key=itemgetter(1), reverse=True)

        for qvar_filler, ep_comb_weight in self._ep_match_combination(reranked_ep_matches_dict):
            # start a new hypothesis
            seed = HypothesisSeed(
                json_graph=self.json_graph,
                core_constraints=core_constraints,
                temporal_constraints=self.temporal_constraints,
                hypothesis=AidaHypothesis(self.json_graph),
                qvar_filler=qvar_filler,
                entrypoints=ep_var_list,
                entrypoint_score=ep_comb_weight)
            seeds_todo.append(seed)

        logging.info(f'Extending {len(seeds_todo)} hypothesis seeds')

        # counter of signatures of query variables, for rank_cutoff
        qs_counter = Counter()

        seed_count = 0

        # extend all hypotheses in the deque until they are done
        while seeds_todo:
            seed_count += 1
            if seed_count % 1000 == 0:
                print(f'Done processing {seed_count} seeds')

            seed = seeds_todo.popleft()

            if self.rank_cutoff is not None:
                qvar_signatures = self._make_qvar_signatures(seed)
                if qvar_signatures is not None:
                    if any(qs_counter[qs] >= self.rank_cutoff for qs in qvar_signatures):
                        # do not process this hypothesis further
                        continue
                    else:
                        for qs in qvar_signatures:
                            qs_counter[qs] += 1

            # we are discarding hypotheses with failed core constraints
            if self.discard_failed_core_constraints:
                # if we have found at least one hypothesis without failed core constraints,
                # discard any new hypothesis with failed core constraints
                if found_hypothesis_wo_failed and not seed.no_failed_core_constraints():
                    continue

            # hypothesis finished.
            if seed.done:
                # if there is no statement in the hypothesis, don't record it
                if not seed.has_statements():
                    continue

                # if this is the first hypothesis without failed core constraints, then remove
                # all previous 'done' hypotheses, as they had failed core constraints
                if self.discard_failed_core_constraints and seed.no_failed_core_constraints():
                    if not found_hypothesis_wo_failed:
                        seeds_done = []

                if seed.no_failed_core_constraints():
                    found_hypothesis_wo_failed = True

                # add typing statements and affiliation statements to the hypothesis seed
                seed.hypothesis_completion()

                # mark this hypothesis as done
                seeds_done.append(seed)

                continue

            # otherwise, extend the current hypothesis
            else:
                news_seeds = seed.extend()
                # put extensions of this hypothesis to the beginning of the queue, such that
                # we explore one hypothesis to the end before we start the next (similar to dfs).
                # this way we can see early if we have hypotheses without failed core constraints
                seeds_todo.extendleft(news_seeds)

        if not found_hypothesis_wo_failed:
            logging.warning('All hypotheses had at least one failed core constraint.')

        # at this point, all hypotheses are as big as they can be.
        return seeds_done

    def _make_qvar_signatures(self, seed: HypothesisSeed):
        def make_one_signature(keys):
            return "_".join(k + "|" + seed.qvar_filler[k][-5:] for k in sorted(keys))

        # if there are less than rank_cutoff_qvar_count non-entrypoint variables, return None
        if len(seed.qvar_filler) - len(seed.entrypoints) < self.rank_cutoff_qvar_count:
            return None

        # make string characterizing entry points
        qs_entry = make_one_signature(seed.entrypoints)
        # and concatenate with string characterizing other fillers
        non_ep_vars = [k for k in seed.qvar_filler.keys() if k not in seed.entrypoints]
        return [qs_entry + "_" + make_one_signature(keys) for keys
                in itertools.combinations(sorted(non_ep_vars), self.rank_cutoff_qvar_count)]

    # ENTRY POINT HANDLING: find any combination of entry point fillers for all the entry points.
    # Yields (qvar_filler, ep_comb_weight) where qvar_filler is a mapping from each entry point
    # variable to a filler, and ep_comb_weight is the confidence of the combination of fillers
    @staticmethod
    def _ep_match_combination(ep_matches_dict: Dict):
        # list of (qvar_filler, ep_comb_weight) tuples
        results = []

        # get all possible combinations of entry point (filler, weight) pairs
        for ep_combination in itertools.product(*ep_matches_dict.values()):
            # a mapping from each ep variable to its filler
            qvar_filler = {}
            # the product of all ep match weights
            ep_comb_weight = 1

            # iterate through each ep variable and the filler and the weight in this combination
            for ep_var, (ep_filler, ep_weight) in zip(ep_matches_dict.keys(), ep_combination):
                qvar_filler[ep_var] = ep_filler
                ep_comb_weight *= ep_weight / 100

            # reject if any two variables are mapped to the same ERE
            if len(set(qvar_filler.values())) != len(qvar_filler.values()):
                continue

            results.append((qvar_filler, ep_comb_weight))

        # sort the results by ep_comb_weight
        for qvar_filler, ep_comb_weight in sorted(results, key=itemgetter(1), reverse=True):
            yield qvar_filler, ep_comb_weight

    # compute the role matching score of an entry point filler, by searching for the number of
    # statements that match core constraints mentioning the entry point
    def _entrypoint_filler_role_score(self, ep_var: str, ep_filler: str, core_constraints: List):
        score = 0

        for _, subj, pred, obj, _ in core_constraints:
            if subj == ep_var:
                # statements adjacent to ep_filler with predicate pred and ep_filler as the subject
                if list(self.json_graph.each_ere_adjacent_stmt(ep_filler, pred, 'subject')):
                    score += 1
            elif obj == ep_var:
                # statements adjacent to ep_filler with predicate pred and ep_filler as the object
                if list(self.json_graph.each_ere_adjacent_stmt(ep_filler, pred, 'object')):
                    score += 1

        return score

    # TEMPORAL ANALYSIS: given the "temporal" piece of a statement of information need,
    # turn the date and time info in the dictionary into Python datetime objects
    # TODO: adapt to new temporal constraint specs
    @staticmethod
    def _pythonize_datetime(temporal_info_dict: Dict):
        aida_time_dict = {}

        for qvar, temporal_constraint in temporal_info_dict.items():
            aida_time_dict[qvar] = {}

            for key in ['start_time', 'end_time']:
                entry = temporal_constraint.get(key, None)
                if entry is not None:
                    aida_time_dict[qvar][key] = AidaIncompleteDate(
                        entry.get('year', None), entry.get('month', None), entry.get('day', None))

        return aida_time_dict
