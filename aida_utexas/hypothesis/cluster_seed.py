"""
Author: Katrin Erk, Mar 2019
- Rule-based creation of initial hypotheses.
- This only adds the statements that the Statement of Information Need asks for, but constructs
all possible cluster seeds that can be made using different statements that all fill the same SOIN.

Update: Pengxiang Cheng, Aug 2020
- Clean-up and refactoring
- Naming conventions:
    - qvar (query variables): variables in query constraints (or, variables in edges of a facet)
    - qvar_filler: a dictionary from each qvar to a filler ERE label
    - ep (entry points): variables grounded by entry point descriptors
    - ep_fillers: a list of possible ERE labels to fill an ep variable, from SoIN matching
"""

import itertools
import logging
import math
from collections import Counter, deque, defaultdict
from operator import itemgetter
from typing import Dict, List, Set

from aida_utexas.aif import JsonGraph
from aida_utexas.hypothesis.aida_hypothesis import AidaHypothesis, AidaHypothesisCollection
from aida_utexas.hypothesis.date_check import AidaIncompleteDate, temporal_constraint_match
from aida_utexas.hypothesis.hypothesis_filter import AidaHypothesisFilter


# The class that holds a single cluster seed: just a data structure, doesn't do much.
class OneClusterSeed:
    def __init__(self, json_graph: JsonGraph, core_constraints: List, temporal_constraints: Dict,
                 hypothesis: AidaHypothesis, qvar_filler: Dict, unfilled: Set = None,
                 unfillable: Set = None, entrypoints: List = None, entrypoint_weight: float = 0.0,
                 penalty_score: float = 0.0):
        # the following data will not not updated, and is kept just for info.
        # the AIDA graph in json format
        self.json_graph = json_graph
        # a list of edge constraints, each being a tuple of (subj, pred, obj)
        self.core_constraints = core_constraints
        # temporal constraints: mapping from query variables to {start_time: ..., end_time:...}
        self.temporal_constraints = temporal_constraints

        # the following data is updated.
        # flags: am I done?
        self.done = False

        # hypothesis is an AidaHypothesis object
        self.hypothesis = hypothesis

        # unfilled, unfillable are indices on self.core_constraints
        self.unfilled = unfilled if unfilled is not None else set(range(len(core_constraints)))
        self.unfillable = unfillable if unfillable is not None else set()

        # hypothesis filter
        self.filter = AidaHypothesisFilter(self.json_graph)

        # mapping from query variables to fillers in json_graph
        self.qvar_filler = qvar_filler

        # entry point variables, which will not be used to rank this hypothesis by novelty later
        self.entrypoints = entrypoints

        # weight from matching the entry points and their corresponding roles in core constraints
        self.entrypoint_weight = entrypoint_weight

        # penalty score for violations in seed creation and extension
        self.penalty_score = penalty_score

        # some penalty constants for things that might go wrong during seed creation and extension
        self.FAILED_QUERY = -0.1
        self.FAILED_TEMPORAL = -0.1
        self.FAILED_ONTOLOGY = -0.1
        self.DUPLICATE_FILLER = -0.01

    # report failed queries of the underlying AidaHypothesis object
    def finalize(self):
        self.hypothesis.add_failed_queries([self.core_constraints[idx] for idx in self.unfillable])
        # self.hypothesis.update_hyp_weight(self.penalty_score)
        self.hypothesis.add_qvar_filler(self.qvar_filler)

        return self.hypothesis

    # extend hypothesis by one statement filling the next fillable core constraint.
    # returns a list of OneClusterSeed objects
    def extend(self):
        nfc = self._next_fillable_constraint()

        if nfc is None:
            # no next fillable constraint.
            # declare done, and return only this object
            self.done = True
            # no change to qvar_filler
            self.unfillable.update(self.unfilled)
            self.unfilled = set()
            return [self]

        elif nfc['failed']:
            # this particular constraint was not fillable, and will never be fillable.
            self.unfilled.remove(nfc['index'])
            self.unfillable.add(nfc['index'])
            # adding failed query penalty
            self.penalty_score += self.FAILED_QUERY
            return [self]

        else:
            # nfc is a structure with entries 'index', 'stmt', 'variable', 'role', etc.
            # find statements that match this constraint, and return a list of extension candidates
            # to match this. these candidates have not been run through the filter yet.
            # format: list of tuples (new_hypothesis, stmt_label, variable, filler)
            ext_candidates = self._extend(nfc)

            if len(ext_candidates) == 0:
                # something has gone wrong
                self.unfilled.remove(nfc['index'])
                self.unfillable.add(nfc['index'])
                # adding failed query penalty
                self.penalty_score += self.FAILED_QUERY
                return [self]

            new_seeds = []
            for new_hypothesis, stmt_label, variable, filler in ext_candidates:
                add_weight = 0

                if self.filter.validate(new_hypothesis, stmt_label):
                    # yes: make a new OneClusterSeed object with this extended hypothesis
                    # TODO: would shallow copy here cause issues?
                    new_qvar_filler = self.qvar_filler.copy()
                    if variable and filler and self.json_graph.is_ere(filler):
                        new_qvar_filler[variable] = filler
                        if filler in self.qvar_filler.values():
                            # some other variable has been mapped to the same ERE,
                            # adding duplicate filler penalty
                            add_weight += self.DUPLICATE_FILLER

                    # changes to unfilled, not to unfillable
                    new_unfilled = self.unfilled.difference([nfc['index']])
                    new_unfillable = self.unfillable.copy()

                    if nfc['relaxed']:
                        # adding failed ontology penalty
                        add_weight += self.FAILED_ONTOLOGY

                    new_seeds.append(OneClusterSeed(
                        self.json_graph, self.core_constraints, self.temporal_constraints,
                        hypothesis=new_hypothesis,
                        qvar_filler=new_qvar_filler,
                        unfilled=new_unfilled,
                        unfillable=new_unfillable,
                        entrypoints=self.entrypoints,
                        entrypoint_weight=self.entrypoint_weight,
                        penalty_score=self.penalty_score + add_weight))

            if len(new_seeds) == 0:
                # all the fillers were filtered away
                self.unfilled.remove(nfc['index'])
                self.unfillable.add(nfc['index'])
                # all candidate statements filtered away, adding failed query penalty
                self.penalty_score += self.FAILED_QUERY
                return [self]
            else:
                return new_seeds

    # return true if there is at least one unfilled core constraint remaining
    def core_constraints_remaining(self):
        return len(self.unfilled) > 0

    # return true if there are no unfillable constraints
    def no_failed_core_constraints(self):
        return len(self.unfillable) == 0

    # return true if the current hypothesis has at least one statement
    def has_statements(self):
        return len(self.hypothesis.stmts) > 0

    # TODO: might define a class for NFC?
    # next fillable constraint from the core constraints list, or None if none fillable
    def _next_fillable_constraint(self):
        # iterate over unfilled query constraints to see if we can find one that can be filled
        for constraint_index in self.unfilled:
            subj, pred, obj = self.core_constraints[constraint_index]

            # if either subj or obj is known (is an ERE or has an entry in qvar_filler,
            # then we should be able to fill this constraint now, or it is unfillable
            subj_filler = self._known_core_constraint_entry(subj)
            obj_filler = self._known_core_constraint_entry(obj)

            if subj_filler is not None and obj_filler is not None:
                # new edge between two known variables
                return self._fill_constraint_two_known_eres(
                    constraint_index, subj_filler, pred, obj_filler)

            elif subj_filler is not None:
                return self._fill_constraint_one_known_ere(
                    constraint_index, subj_filler, 'subject', pred, obj, 'object')

            elif obj_filler is not None:
                return self._fill_constraint_one_known_ere(
                    constraint_index, obj_filler, 'object', pred, subj, 'subject')

            else:
                # this constraint cannot be filled at this point,
                # wait and see if it can be filled some other time
                continue

        # reaching this point, and not having returned anything:
        # this means we do not have any fillable constraints left
        return None

    # given a subject/object from a core constraint, if it is an ERE from the graph, or a variable
    # for which we already know the filler, return the filler ERE ID, otherwise return None.
    def _known_core_constraint_entry(self, entry):
        if self.json_graph.has_node(entry):
            return entry
        elif entry in self.qvar_filler:
            return self.qvar_filler[entry]
        else:
            return None

    # see if this label can be generalized by cutting out the lowest level of specificity.
    @staticmethod
    def _generalize_pred(pred_label):
        pieces = pred_label.split('_')
        if len(pieces) == 1:
            pred_class = pred_label
            pred_role = ''
        elif len(pieces) == 2:
            pred_class = pieces[0]
            pred_role = pieces[1]
        else:
            logging.warning(f'unexpected number of underscores in {pred_label}, could not split')
            return None

        pieces = pred_class.split('.')
        if len(pieces) <= 2:
            # no more general class possible
            return None

        # we can try a more lenient match
        return '{}_{}'.format('.'.join(pieces[:-1]), pred_role)

    # returns list of statement candidates that have ERE as the 'role' (subject, object) and
    # have predicate 'pred', also returns whether the statements had to be relaxed.
    # If no statements could be found, returns None.
    def _statement_candidates(self, ere, pred, role):
        candidates = list(self.json_graph.each_ere_adjacent_stmt(
            ere_label=ere, predicate=pred, ere_role=role))

        # success, we found some statements with strictly matched predicate
        if len(candidates) > 0:
            return candidates, False

        # no success, see if more lenient match will work
        lenient_pred = self._generalize_pred(pred)

        if lenient_pred is not None:
            # try the more general class
            candidates = list(self.json_graph.each_ere_adjacent_stmt(
                ere_label=ere, predicate=lenient_pred, ere_role=role))

            if len(candidates) > 0:
                # success, we found some statement with leniently matched predicate
                return candidates, True

        # still no success, return None
        return None, None

    # try to fill this constraint from the graph, either strictly or leniently.
    # one side of this constraint is a known ERE, the other side can be anything
    def _fill_constraint_one_known_ere(self, constraint_index, known_ere, known_role, pred,
                                       unknown, unknown_role):
        # find statements that could potentially fill the role
        candidates, relaxed = self._statement_candidates(known_ere, pred, known_role)

        if candidates is None:
            # no candidates found at all, constraint is unfillable
            return {'index': constraint_index, 'failed': True}

        # check if unknown is a constant in the graph, in which case it is not really unknown
        if self._is_string_constant(unknown):
            # which of the statement candidates have the right filler?
            candidates = [s for s in candidates
                          if self.json_graph.node_dict[s][unknown_role] == unknown]
            if len(candidates) == 0:
                return {'index': constraint_index, 'failed': True}
            else:
                return {
                    'index': constraint_index,
                    'stmt': candidates,
                    'role': known_role,
                    'has_variable': False,
                    'relaxed': relaxed,
                    'failed': False
                }

        else:
            # nope, we have a variable we can fill, any fillers?
            return {
                'index': constraint_index,
                'stmt': candidates,
                'variable': unknown,
                'role': known_role,
                'has_variable': True,
                'relaxed': relaxed,
                'failed': False
            }

    # try to fill this constraint from the graph, either strictly or leniently.
    # both sides of this constraint are known EREs
    def _fill_constraint_two_known_eres(self, constraint_index, subj_ere, pred, obj_ere):
        # find statements that could fill the subject role
        possible_candidates, relaxed = self._statement_candidates(subj_ere, pred, 'subject')

        if possible_candidates is None:
            # no candidates found at all, constraint is unfillable
            return {'index': constraint_index, 'failed': True}

        # we did find candidates. check whether any of the candidates has obj_ere as its object
        candidates = [c for c in possible_candidates if self.json_graph.stmt_object(c) == obj_ere]
        if len(candidates) == 0:
            # constraint is unfillable
            return {'index': constraint_index, 'failed': True}

        else:
            return {'index': constraint_index,
                    'failed': False,
                    'stmt': candidates,
                    'has_variable': False,
                    'relaxed': relaxed}

    # nfc is a structure with entries 'index', 'stmt', 'variable', 'role', etc.
    # find statements that match this constraint, and return a list of tuples:
    # (new_hypothesis, stmt_label, variable, filler)
    def _extend(self, nfc):

        # did not find any matches to this constraint
        if len(nfc['stmt']) == 0:
            return []

        # this next fillable constraint states a constant string value about a known ERE,
        # or it states a new connection between known EREs. If we do have more than one
        # matching statements. add just the first one, they are identical
        if not nfc['has_variable']:
            stmt_label = nfc['stmt'][0]
            if not self.json_graph.has_node(stmt_label):
                logging.warning(f'Statement {stmt_label} returned by _next_fillable_constraint '
                                f'not found in the graph.')
                return []

            else:
                # can this statement be added to the hypothesis without contradiction?
                # extended hypothesis
                return [(self.hypothesis.extend(stmt_label, core=True), stmt_label, None, None)]

        # we have a new variable (nfc['variable']), and its role (nfc['role'])
        # find all statements that can fill the current constraint of the variable and the role
        # if we don't find anything, re-run with more leeway on temporal constraints
        ext_candidates, has_temporal_constraint = self._extend_with_variable(nfc, leeway=0)
        if len(ext_candidates) == 0 and has_temporal_constraint:
            # adding penalty for relaxing of temporal constraint
            self.penalty_score += self.FAILED_TEMPORAL
            # relax temporal matching by one day
            ext_candidates, has_temporal_constraint = self._extend_with_variable(nfc, leeway=1)

        if len(ext_candidates) == 0 and has_temporal_constraint:
            # adding penalty for relaxing of temporal constraint
            self.penalty_score += self.FAILED_TEMPORAL
            # relax temporal matching: everything goes
            ext_candidates, has_temporal_constraint = self._extend_with_variable(nfc, leeway=2)

        return ext_candidates

    # nfc is a structure with entries 'index', 'stmt', 'variable', 'role', etc.
    # find statements that match this constraint, and return a pair (hyp, has_temporal_constraint)
    # where hyp is a list of tuples; (new_hypothesis, stmt_label, variable, filler)
    # and has_temporal_constraint is true if there was at least one temporal constraint violated
    def _extend_with_variable(self, nfc, leeway=0):

        ext_candidates = []
        has_temporal_constraint = False

        other_role = self._nfc_other_role(nfc)
        if other_role is None:
            # if nfc does not have 'role', we should be in this method, there must be some error
            return ext_candidates, has_temporal_constraint

        for stmt_label in nfc['stmt']:
            if not self.json_graph.has_node(stmt_label):
                logging.warning(f'Statement {stmt_label} returned by _next_fillable_constraint '
                                f'not found in the graph.')
                continue

            # determine the ERE or value that fills the role that has the variable
            filler = getattr(self.json_graph.node_dict[stmt_label], other_role)

            # is this an ERE? if so, we need to check for temporal constraints.
            if self.json_graph.is_ere(filler):
                # TODO: maybe check second_constraint_violated before temporal_constraint?
                # is there a problem with a temporal constraint?
                if not temporal_constraint_match(
                        self.json_graph.node_dict[filler],
                        self.temporal_constraints.get(nfc['variable'], None), leeway):
                    # if this filler runs afoul of some temporal constraint, do not use it
                    has_temporal_constraint = True
                    continue

                # we also check whether including this statement will violate another constraint.
                # if so, we do not include it
                if self._second_constraint_violated(nfc['variable'], filler, nfc['index']):
                    continue

            # can this statement be added to the hypothesis without contradiction?
            # extended hypothesis
            new_hypothesis = self.hypothesis.extend(stmt_label, core=True)
            ext_candidates.append((new_hypothesis, stmt_label, nfc['variable'], filler))

        return ext_candidates, has_temporal_constraint

    # second constraint violated: given a variable and its filler, see if filling this qvar with
    # this filler will make any constraint that is yet unfilled unfillable
    def _second_constraint_violated(self, variable, filler, except_index):
        for constraint_index in self.unfilled:
            if constraint_index == except_index:
                # this was the constraint we were just going to fill, don't re-check it
                continue

            subj, pred, obj = self.core_constraints[constraint_index]

            if subj == variable and obj in self.qvar_filler:
                candidates, _ = self._statement_candidates(filler, pred, 'subject')
                # there is not statement with pred as predicate and filler as subject
                if candidates is None:
                    return True
                else:
                    candidates = [c for c in candidates
                                  if self.json_graph.stmt_object(c) == self.qvar_filler[obj]]
                    # there is no statement with pred as predicate, filler as subject, and
                    # the already filled obj as the object
                    if len(candidates) == 0:
                        return True

            # found a constraint involving this variable as object and another variable that
            # has already been filled as subject
            elif obj == variable and subj in self.qvar_filler:
                candidates = self._statement_candidates(filler, pred, 'object')
                # there is not statement with pred as predicate and filler as object
                if candidates is None:
                    return True
                else:
                    candidates = [c for c in candidates
                                  if self.json_graph.stmt_subject(c) == self.qvar_filler[subj]]
                    # there is no statement with pred as predicate, filler as object, and
                    # the already filled obj as the object
                    if len(candidates) == 0:
                        return True

        return False

    # given a next_fillable_constraint dictionary,
    # if it has a role of 'subject' return 'object' and vice versa
    @staticmethod
    def _nfc_other_role(nfc):
        if nfc['role'] == 'subject':
            return 'object'
        elif nfc['role'] == 'object':
            return 'subject'
        else:
            print('ClusterSeed error: unknown role', nfc['role'])
            return None

    # is the given string a variable, or should it be viewed as a string constant?
    # use the list of all string constants in the given graph
    def _is_string_constant(self, node_label):
        return node_label in self.json_graph.string_constants


# class that manages all cluster seeds
class ClusterSeeds:
    # initialize with a JsonGraph object and json representation of a query
    def __init__(self, json_graph: JsonGraph, query_json: dict,
                 discard_failed_queries: bool = False, early_cutoff: int = None,
                 qs_cutoff: int = None):
        self.json_graph = json_graph
        self.query_json = query_json

        # discard queries with any failed constraints?
        self.discard_failed_queries = discard_failed_queries
        # cut off after early_cutoff entry point combinations?
        self.early_cutoff = early_cutoff
        # cut off partially formed hypotheses during creation
        # if there are at least qs_cutoff other hypothesis seeds
        # with the same fillers for qs_cutoff_count query variables?
        # TODO: rename it to rank_cutoff?
        self.qs_cutoff = qs_cutoff
        self.qs_cutoff_count = 3

        # parameters for ranking
        self.rank_first_k = 100
        self.bonus_for_novelty = 5
        self.consider_next_k_in_reranking = 10000

        # make seed clusters
        self.seeds = self._make_seeds()

    # export hypotheses to AidaHypothesisCollection
    def finalize(self):
        # rank the done cluster seeds
        logging.info('Ranking cluster seeds')
        ranked_seeds = self._rank_seeds()

        hypotheses_to_export = []

        # turn ranks into the log weights of seed hypotheses
        # meaningless numbers. just assign 1/2, 1/3, 1/4, ...
        for rank, seed in enumerate(ranked_seeds):
            seed.hypothesis.update_weight(math.log(1.0 / (rank + 1)))
            hypotheses_to_export.append(seed.finalize())

        return AidaHypothesisCollection(hypotheses_to_export)

    # HYPOTHESIS SEED CREATION: create initial cluster seeds. This is called from __init__
    def _make_seeds(self) -> List[OneClusterSeed]:
        # the queue of seeds-in-making
        seeds_todo = deque()
        # the list of finished seeds
        seeds_done = []

        # have we found any hypothesis without failed queries yet?
        # if so, we can eliminate all hypotheses with failed queries
        found_hypothesis_wo_failed_queries = False

        # early_cutoff per facet
        # facet_cutoff = self.early_cutoff / len(self.query_json['facets']) \
        #     if self.early_cutoff is not None else None

        all_ep_fillers = self.query_json['entrypoints']
        all_ep_weights = self.query_json['entrypointWeights']

        logging.info('Initializing cluster seeds (if stalled, set early_cutoff)')

        for facet in self.query_json['facets']:
            # list of query constraints in this facet
            query_constraints = facet['queryConstraints']

            # variables occurring in this facet: query constraints have the form [subj, pred, obj],
            # where subj, obj are variables.
            facet_variables = set(c[0] for c in query_constraints).union(
                c[2] for c in query_constraints)

            # variables to fill: all entry points that appear in the query constraints of this facet
            facet_ep_variables = sorted(e for e in all_ep_fillers.keys() if e in facet_variables)

            # entry point fillers and weights filtered and reranked by both ep match scores and
            # role match scores
            reranked_all_ep_fillers = {}
            reranked_all_ep_weights = {}

            for ep_var in facet_ep_variables:
                ep_fillers = all_ep_fillers[ep_var]
                ep_weights = all_ep_weights[ep_var]

                logging.info('Entry point: {}'.format(ep_var))

                ep_weights_w_role = {}

                fillers_filtered_both = []
                fillers_filtered_role_score = []

                for ep_filler, ep_weight in zip(ep_fillers, ep_weights):
                    ep_role_score = self._entrypoint_filler_role_score(ep_var, ep_filler, facet)
                    ep_weights_w_role[ep_filler] = (ep_weight, ep_role_score)
                    # an ep_role_score > 0 means there is at least one role match for the filler
                    if ep_role_score > 0:
                        fillers_filtered_role_score.append(ep_filler)
                        # an ep_weight > 50 is considered as a "good" match (max score = 100)
                        if ep_weight > 50.0:
                            fillers_filtered_both.append(ep_filler)

                if len(fillers_filtered_both) > 0:
                    logging.info(f'Kept {len(fillers_filtered_both)} fillers with both '
                                 f'SoIN weight > 50 and role score > 0')
                    fillers_to_keep = [(f, ep_weights_w_role[f][0] * ep_weights_w_role[f][1])
                                       for f in fillers_filtered_both]
                elif len(fillers_filtered_role_score) > 0:
                    logging.info(f'Kept {len(fillers_filtered_role_score)} fillers with '
                                 f'role score > 0')
                    fillers_to_keep = [(f, ep_weights_w_role[f][0] * ep_weights_w_role[f][1])
                                       for f in fillers_filtered_role_score]
                else:
                    logging.info(f'Kept all {len(ep_fillers)} fillers (there is no filler with '
                                 f'role score > 0)')
                    fillers_to_keep = list(zip(ep_fillers, ep_weights))

                fillers_to_keep.sort(key=itemgetter(1), reverse=True)
                reranked_all_ep_fillers[ep_var] = [f for f, w in fillers_to_keep]
                reranked_all_ep_weights[ep_var] = [w for f, w in fillers_to_keep]

            if self.early_cutoff is None:
                ep_combinations = self._each_entry_point_combination(
                    reranked_all_ep_fillers, reranked_all_ep_weights)
            else:
                ep_combinations = self._each_entry_point_combination_w_early_cutoff(
                    reranked_all_ep_fillers, reranked_all_ep_weights, self.early_cutoff)

            for qvar_filler, ep_comb_weight in ep_combinations:
                # start a new hypothesis
                seed = OneClusterSeed(
                    json_graph=self.json_graph,
                    core_constraints=facet['queryConstraints'],
                    temporal_constraints=self._pythonize_datetime(facet.get('temporal', {})),
                    hypothesis=AidaHypothesis(self.json_graph),
                    qvar_filler=qvar_filler,
                    entrypoints=list(qvar_filler.keys()),
                    entrypoint_weight=ep_comb_weight)
                seeds_todo.append(seed)

        logging.info('Extending cluster seeds (if too many, reduce rank_cutoff)')

        # counter of signatures of query variables, for qs_cutoff
        qs_counter = Counter()

        seed_count = 0

        # extend all hypotheses in the deque until they are done
        while seeds_todo:
            seed_count += 1
            if seed_count % 1000 == 0:
                logging.info(f'Done processing {seed_count} seeds')

            seed = seeds_todo.popleft()

            if self.qs_cutoff is not None:
                qvar_signatures = self._make_qvar_signatures(seed)
                if qvar_signatures is not None:
                    if any(qs_counter[qs] >= self.qs_cutoff for qs in qvar_signatures):
                        # do not process this hypothesis further
                        continue
                    else:
                        for qs in qvar_signatures:
                            qs_counter[qs] += 1

            # we are discarding hypotheses with failed queries
            if self.discard_failed_queries:
                # if we have found at least one hypothesis without failed queries,
                # discard any new hypothesis with failed queries
                if found_hypothesis_wo_failed_queries and not seed.no_failed_core_constraints():
                    continue

            # hypothesis finished.
            if seed.done:
                # if there is no statement in the hypothesis, don't record it
                if not seed.has_statements():
                    continue

                # if this is the first hypothesis without failed queries, then remove all previous
                # 'done' hypotheses, as they had failed queries
                if self.discard_failed_queries and seed.no_failed_core_constraints():
                    if not found_hypothesis_wo_failed_queries:
                        seeds_done = []

                if seed.no_failed_core_constraints():
                    found_hypothesis_wo_failed_queries = True

                # mark this hypothesis as done
                seeds_done.append(seed)

                continue

            # otherwise, extend the current hypothesis
            else:
                news_seeds = seed.extend()
                # put extensions of this hypothesis to the beginning of the queue, such that
                # we explore one hypothesis to the end before we start the next (similar to dfs).
                # this way we can see early if we have hypotheses without failed queries
                seeds_todo.extendleft(news_seeds)

        if not found_hypothesis_wo_failed_queries:
            logging.warning('All hypotheses had at least one failed query.')

        # at this point, all hypotheses are as big as they can be.
        return seeds_done

    def _make_qvar_signatures(self, seed: OneClusterSeed):
        def make_one_signature(keys):
            return "_".join(k + "|" + seed.qvar_filler[k][-5:] for k in sorted(keys))

        # if there are less than qs_cutoff_count non-entrypoint variables, return None
        if len(seed.qvar_filler) - len(seed.entrypoints) < self.qs_cutoff_count:
            return None

        # make string characterizing entry points
        qs_entry = make_one_signature(seed.entrypoints)
        # and concatenate with string characterizing other fillers
        non_ep_vars = [k for k in seed.qvar_filler.keys() if k not in seed.entrypoints]
        return [qs_entry + "_" + make_one_signature(keys)
                for keys in itertools.combinations(sorted(non_ep_vars), self.qs_cutoff_count)]

    # ENTRY POINT HANDLING: find any combination of entry point fillers for all the entry points.
    # Yields (qvar_filler, ep_comb_weight) where qvar_filler is a mapping from each entry point
    # variable to a filler, and ep_comb_weight is the confidence of the combination of fillers
    @staticmethod
    def _each_entry_point_combination(all_ep_fillers: Dict, all_ep_weights: Dict):

        all_qvar_fillers, all_ep_comb_weights = [], []

        # here we do a itertools.product of entry point filler indices, so we can access
        # each filler as well as its weight
        for filler_indices in itertools.product(*(range(len(ep_fillers)) for ep_fillers
                                                  in all_ep_fillers.values())):
            # ep variable -> filler mapping: pair each entry point variable with the i-th filler,
            # where i is the filler index for that entry point variable
            qvar_filler = {v: all_ep_fillers[v][i] for v, i in zip(all_ep_fillers, filler_indices)}

            # reject if any two variables are mapped to the same ERE
            if len(set(qvar_filler.values())) != len(qvar_filler.values()):
                continue

            all_qvar_fillers.append(qvar_filler)

            # we multiple the weights of all fillers to get the combination weight
            ep_comb_weight = 1
            for v, i in zip(all_ep_weights, filler_indices):
                ep_comb_weight *= all_ep_weights[v][i]
            all_ep_comb_weights.append(ep_comb_weight)

        for qvar_filler, ep_comb_weight in sorted(zip(all_qvar_fillers, all_ep_comb_weights),
                                                  key=itemgetter(1), reverse=True):
            yield qvar_filler, ep_comb_weight

    # ENTRY POINT HANDLING: in the context of early_cutoff.
    # Note that we use a ad-hoc complex logic here, which is probably not needed in M36 evaluation.
    @staticmethod
    def _each_entry_point_combination_w_early_cutoff(all_ep_fillers: Dict, all_ep_weights: Dict,
                                                     early_cutoff: int):
        # for each entry point variable, build a mapping from each weight value to all fillers with
        # that weight value
        ep_weight_filler_mapping = {}

        for ep_var, ep_fillers in all_ep_fillers.items():
            # mapping from each weight value to corresponding fillers
            weight_to_fillers = defaultdict(list)

            ep_weights = all_ep_weights[ep_var]

            # Make sure all weights are positive (in case of using role weighting in SoIN matching,
            # there might be negative weights for entry points, which might mess up rankings)
            # NOT NEEDED ANYMORE: we are not using role weights in SoIN matching
            # if min(ep_weights) < 0.1:
            #     ep_weights = [w - min(ep_weights) + 0.1 for w in ep_weights]
            for ep_filler, ep_weight in zip(ep_fillers, ep_weights):
                weight_to_fillers[ep_weight].append(ep_filler)

            # sort by weight from high to low
            ep_weight_filler_mapping[ep_var] = dict(
                sorted(weight_to_fillers.items(), key=itemgetter(0), reverse=True))

        # helper function: with a mapping from each ep variable to a weight value, find a list of
        # qvar_filler mappings corresponding to the weights
        def _ep_weights_to_qvar_fillers(_ep_weights: Dict):
            _fillers_list = []
            for _ep_var, _weight in _ep_weights.items():
                _fillers_list.append(ep_weight_filler_mapping[_ep_var][_weight])
            _qvar_filler_list = []
            for _fillers in itertools.product(*_fillers_list):
                # Filter cases where there are duplicate node ids for different entry points
                if len(set(_fillers)) == len(_fillers):
                    _qvar_filler_list.append({_ep_var: _filler for _ep_var, _filler
                                              in zip(_ep_weights.keys(), _fillers)})
            return _qvar_filler_list

        all_qvar_fillers = []
        all_ep_comb_weights = []

        # helper function: with a list of ep_weights mappings and a list of combination weights,
        # add corresponding qvar_fillers
        def _process(_ep_weights_list: List[Dict], _comb_weight_list: List[float], group_idx: int):
            for _ep_weights, _comb_weight in sorted(zip(_ep_weights_list, _comb_weight_list),
                                                    key=itemgetter(1), reverse=True):
                _qvar_fillers = _ep_weights_to_qvar_fillers(_ep_weights)

                print('Found {} filler combinations with weight {} (group #{})'.format(
                    len(_qvar_fillers), _comb_weight, group_idx))

                all_qvar_fillers.extend(_qvar_fillers)
                all_ep_comb_weights.extend([_comb_weight] * len(_qvar_fillers))

                # TODO: do we need this condition here?
                if len(all_ep_comb_weights) >= early_cutoff:
                    break

        # Group 1 (all highest)
        g1_ep_weights = {}
        g1_comb_weight = 1.0
        for ep_var, weight_filler_mapping in ep_weight_filler_mapping.items():
            var_weight = list(weight_filler_mapping.keys())[0]
            g1_ep_weights[ep_var] = var_weight
            g1_comb_weight *= var_weight

        _process([g1_ep_weights], [g1_comb_weight], group_idx=1)

        # Group 2 (all-but-one highest & one second-highest)
        if len(all_qvar_fillers) < early_cutoff:
            g2_ep_weights_list = []
            g2_comb_weight_list = []

            # For each entry point, select its second-highest weighted fillers
            for idx in range(len(ep_weight_filler_mapping)):
                ep_weights, comb_weight = {}, 1.0

                for ep_var_idx, (ep_var, weight_filler_mapping) in enumerate(
                        ep_weight_filler_mapping.items()):
                    if ep_var_idx == idx:
                        if len(weight_filler_mapping) < 2:
                            continue
                        var_weight = list(weight_filler_mapping.keys())[1]
                    else:
                        var_weight = list(weight_filler_mapping.keys())[0]

                    ep_weights[ep_var] = var_weight
                    comb_weight *= var_weight

                if len(ep_weights) == len(ep_weight_filler_mapping):
                    g2_ep_weights_list.append(ep_weights)
                    g2_comb_weight_list.append(comb_weight)

            _process(g2_ep_weights_list, g2_comb_weight_list, group_idx=2)

        # Group 3 (all-but-one highest & one third-highest,
        # or all-but-two highest & two second-highest)
        if len(all_qvar_fillers) < early_cutoff:
            g3_ep_weights_list = []
            g3_comb_weight_list = []

            # For each entry point, select its third-highest weighted fillers
            for idx in range(len(ep_weight_filler_mapping)):
                ep_weights, comb_weight = {}, 1.0

                for ep_var_idx, (ep_var, weight_filler_mapping) in enumerate(
                        ep_weight_filler_mapping.items()):
                    if ep_var_idx == idx:
                        if len(weight_filler_mapping) < 3:
                            continue
                        var_weight = list(weight_filler_mapping.keys())[2]
                    else:
                        var_weight = list(weight_filler_mapping.keys())[0]

                    ep_weights[ep_var] = var_weight
                    comb_weight *= var_weight

                if len(ep_weights) == len(ep_weight_filler_mapping):
                    g3_ep_weights_list.append(ep_weights)
                    g3_comb_weight_list.append(comb_weight)

            # For each combination of 2 entry points, select their second-highest weighted fillers
            for i1, i2 in itertools.combinations(range(len(ep_weight_filler_mapping)), 2):
                ep_weights, comb_weight = {}, 1.0

                for ep_var_idx, (ep_var, weight_filler_mapping) in enumerate(
                        ep_weight_filler_mapping.items()):
                    if ep_var_idx == i1 or ep_var_idx == i2:
                        if len(weight_filler_mapping) < 2:
                            continue
                        var_weight = list(weight_filler_mapping.keys())[1]
                    else:
                        var_weight = list(weight_filler_mapping.keys())[0]

                    ep_weights[ep_var] = var_weight
                    comb_weight *= var_weight

                if len(ep_weights) == len(ep_weight_filler_mapping):
                    g3_ep_weights_list.append(ep_weights)
                    g3_comb_weight_list.append(comb_weight)

            _process(g3_ep_weights_list, g3_comb_weight_list, group_idx=3)

        # Group 4 (all-but-one highest & one forth-highest,
        # or all-but-two highest & one second-highest & one third-highest,
        # or all-but-three highest & three second-highest)
        if len(all_qvar_fillers) < early_cutoff:
            g4_ep_weights_list = []
            g4_comb_weight_list = []

            # For each entry point, select its forth-highest weighted fillers
            for idx in range(len(ep_weight_filler_mapping)):
                ep_weights, comb_weight = {}, 1.0

                for ep_var_idx, (ep_var, weight_filler_mapping) in enumerate(
                        ep_weight_filler_mapping.items()):
                    if ep_var_idx == idx:
                        if len(weight_filler_mapping) < 4:
                            continue
                        var_weight = list(weight_filler_mapping.keys())[3]
                    else:
                        var_weight = list(weight_filler_mapping.keys())[0]

                    ep_weights[ep_var] = var_weight
                    comb_weight *= var_weight

                if len(ep_weights) == len(ep_weight_filler_mapping):
                    g4_ep_weights_list.append(ep_weights)
                    g4_comb_weight_list.append(comb_weight)

            # For each permutation of 2 entry points, select the third-highest weighted fillers for
            # one of them, and the second-highest weighted fillers for the other
            for i1, i2 in itertools.permutations(range(len(ep_weight_filler_mapping)), 2):
                ep_weights, comb_weight = {}, 1.0

                for ep_var_idx, (ep_var, weight_filler_mapping) in enumerate(
                        ep_weight_filler_mapping.items()):
                    if ep_var_idx == i1:
                        if len(weight_filler_mapping) < 3:
                            continue
                        var_weight = list(weight_filler_mapping.keys())[2]
                    elif ep_var_idx == i2:
                        if len(weight_filler_mapping) < 2:
                            continue
                        var_weight = list(weight_filler_mapping.keys())[1]
                    else:
                        var_weight = list(weight_filler_mapping.keys())[0]

                    ep_weights[ep_var] = var_weight
                    comb_weight *= var_weight

                if len(ep_weights) == len(ep_weight_filler_mapping):
                    g4_ep_weights_list.append(ep_weights)
                    g4_comb_weight_list.append(comb_weight)

            # For each combination of 3 entry points, select their second-highest weighted fillers
            for i1, i2, i3 in itertools.combinations(range(len(ep_weight_filler_mapping)), 3):
                ep_weights, comb_weight = {}, 1.0

                for ep_var_idx, (ep_var, weight_filler_mapping) in enumerate(
                        ep_weight_filler_mapping.items()):
                    if ep_var_idx == i1 or ep_var_idx == i2 or ep_var_idx == i3:
                        if len(weight_filler_mapping) < 2:
                            continue
                        var_weight = list(weight_filler_mapping.keys())[1]
                    else:
                        var_weight = list(weight_filler_mapping.keys())[0]

                    ep_weights[ep_var] = var_weight
                    comb_weight *= var_weight

                if len(ep_weights) == len(ep_weight_filler_mapping):
                    g4_ep_weights_list.append(ep_weights)
                    g4_comb_weight_list.append(comb_weight)

            _process(g4_ep_weights_list, g4_comb_weight_list, group_idx=4)

        for qvar_filler, ep_comb_weight in sorted(zip(all_qvar_fillers, all_ep_comb_weights),
                                                  key=itemgetter(1), reverse=True):
            yield qvar_filler, ep_comb_weight

    # compute the role matching score of an entry point filler, by searching for the number of
    # statements that match query constraints mentioning the entry point
    def _entrypoint_filler_role_score(self, ep_var: str, ep_filler: str, facet: Dict):
        score = 0

        for subj, pred, obj in facet['queryConstraints']:
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

    # RANKING: rank all cluster seeds in self.seeds
    # first we group all seeds by entry point weight and penalty score, then sort seeds in each
    # group by connectedness and novelty
    def _rank_seeds(self):
        seeds_by_weight = defaultdict(list)
        for s in self.seeds:
            seeds_by_weight[(s.entrypoint_weight, s.penalty_score)].append(s)

        ranked_seeds = []
        for _, seeds in sorted(seeds_by_weight.items(), key=itemgetter(0), reverse=True):
            ranked_seeds += self._rank_seeds_by_novelty(self._rank_seeds_by_connectedness(seeds))

        return ranked_seeds

    def _rank_seeds_by_connectedness(self, seeds: List[OneClusterSeed]):
        if len(seeds) == 0:
            return seeds

        scores = []

        for seed in seeds:
            out_deg = 0

            # for each ERE of this hypothesis
            for ere_label in seed.hypothesis.eres():
                # count the number of statements adjacent to the ERE
                out_deg += len(list(self.json_graph.each_ere_adjacent_stmt(ere_label)))

            scores.append(out_deg)

        return [seed for seed, w in sorted(zip(seeds, scores), key=itemgetter(1), reverse=True)]

    def _rank_seeds_by_novelty(self, seeds: List[OneClusterSeed]):
        if len(seeds) == 0:
            return seeds

        # the characterization of query variable fillers for already ranked seeds.
        # format: a mapping from each qvar to a counter of fillers
        # we penalize a seed that has the same qvar filler that we have seen before, with a value
        # equivalent to the number of previous seeds that had the same filler.
        qvar_characterization = defaultdict(Counter)

        # update qvar_characterization by counting the qvar fillers of a new cluster seed
        def _update_qvar_characterization(_seed: OneClusterSeed):
            for _qvar, _filler in _seed.qvar_filler.items():
                # do not include entry points: novelty in entry points is not rewarded
                if _qvar not in _seed.entrypoints:
                    qvar_characterization[_qvar][_filler] += 1

        # seeds already ranked, initializing with the first seed
        seeds_done = [seeds[0]]
        # seeds to rank
        seeds_todo = seeds[1:]

        _update_qvar_characterization(seeds[0])

        while len(seeds_todo) > max(0, len(seeds) - self.rank_first_k):
            # choose the next most novel seed from seeds_todo
            next_seed_idx = self._choose_next_novel_seed(seeds_todo, qvar_characterization)
            if next_seed_idx is None:
                # we didn't find any more items to rank
                break

            # append the next best item to the seeds_done
            next_seed = seeds_todo.pop(next_seed_idx)
            seeds_done.append(next_seed)

            _update_qvar_characterization(next_seed)

        # at this point we have ranked the self.rank_first_k items
        # just attach the rest of the items at the end
        return seeds_done + seeds_todo

    def _choose_next_novel_seed(self, seeds: List[OneClusterSeed], qvar_characterization: Dict):
        best_index, best_score = None, float('-inf')

        # for each seed, determine its novelty score by comparing to qvar_characterization
        for index, seed in enumerate(seeds):
            if index >= self.consider_next_k_in_reranking:
                # we have run out of the next k to consider, don't go further down the list
                break

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
                    score += self.bonus_for_novelty

            # at this point we have the score for the current seed.
            # if it is the maximally achievable score, stop here and go with this seed
            if score >= self.bonus_for_novelty * len(qvar_characterization):
                best_index = index
                break

            # if the score is better than the previous best, record this index as the best one
            if score > best_score:
                best_index = index
                best_score = score

        return best_index
