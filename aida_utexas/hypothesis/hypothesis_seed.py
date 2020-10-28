"""
Author: Katrin Erk, Mar 2019
- Rule-based creation of initial hypotheses.
- This only adds the statements that the Statement of Information Need asks for, but constructs all
possible hypothesis seeds that can be made using different statements that all fill the same SOIN.

Update: Pengxiang Cheng, Aug 2020
- Clean-up and refactoring
- Naming conventions:
    - qvar (query variables): variables in core constraints (edges of a facet)
    - qvar_filler: a dictionary from each qvar to a filler ERE label
    - ep (entry points): variables grounded by entry point descriptors
    - ep_fillers: a list of possible ERE labels to fill an ep variable, from SoIN matching

Update: Pengxiang Cheng, Sep 2020
- Split cluster_seeds.py into two separate files, and rename the classes
- Use a dataclass for NextFillableConstraint to make it more readable
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from aida_utexas.aif import JsonGraph
from aida_utexas.hypothesis.aida_hypothesis import AidaHypothesis
from aida_utexas.hypothesis.date_check import temporal_constraint_match, AidaIncompleteDate
from aida_utexas.hypothesis.hypothesis_filter import AidaHypothesisFilter


@dataclass
class NextFillableConstraint:
    # the index of the constraint
    constraint_index: int
    # whether this constraint is fillable or not
    failed: bool
    # a list of candidate statements that can potentially fill the constraint
    candidate_stmts: List[str] = field(default_factory=list)
    # whether we use a relaxed ontology to match the edge label
    relaxed_ontology: bool = False
    # whether there is an open variable to fill
    has_variable: bool = False
    # the name of the open variable
    open_variable: str = None
    # the role of the open variable, can only be subject or object
    open_role: str = None


@dataclass
class ExtensionCandidate:
    # the new hypothesis
    new_hypothesis: AidaHypothesis
    # the label of the new statement
    stmt_label: str
    # the new query variable to fill, optional
    variable: str = None
    # the filler to the new query variable, optional
    filler: str = None
    # the penalty score from objType violation, optional (default 0)
    obj_type_penalty: float = 0.0


# The class that holds a single hypothesis seed: just a data structure, doesn't do much.
class HypothesisSeed:
    # some penalty constants for things that might go wrong during seed creation and extension
    FAILED_CORE = -0.1
    FAILED_TEMPORAL = -0.1
    FAILED_ONTOLOGY = -0.1
    DUPLICATE_FILLER = -0.01
    FAILED_OBJ_TYPE = -0.1

    def __init__(self, json_graph: JsonGraph, core_constraints: List, temporal_constraints: Dict,
                 hypothesis: AidaHypothesis, qvar_filler: Dict, unfilled: Set = None,
                 unfillable: Set = None, entrypoints: List = None, entrypoint_score: float = 0.0,
                 penalty_score: float = 0.0, connectedness_score: float = None,
                 plausibility_score: float = None):
        # the following data will not not updated, and is kept just for info.
        # the AIDA graph in json format
        self.json_graph = json_graph
        # a list of edge constraints, each being a tuple of (subj, pred, obj, obj_type)
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
        self.entrypoint_score = entrypoint_score

        # penalty score for violations in seed creation and extension
        self.penalty_score = penalty_score

        # connectedness score of the hypothesis seed (average number of adjacent statements of EREs)
        self.connectedness_score = connectedness_score

        # plausibility score of the hypothesis seed
        self.plausibility_score = plausibility_score

    def to_json(self):
        temporal_constraints_json = {}
        for qvar, temporal_constraint in self.temporal_constraints.items():
            temporal_constraints_json[qvar] = {
                key: val.to_json() for key, val in temporal_constraint.items()}

        return {
            'hypothesis': self.hypothesis.to_json(),
            'core_constraints': self.core_constraints,
            'temporal_constraints': temporal_constraints_json,
            'unfilled': list(self.unfilled),
            'unfillable': list(self.unfillable),
            'qvar_filler': self.qvar_filler,
            'entrypoints': self.entrypoints,
            'entrypoint_score': self.entrypoint_score,
            'penalty_score': self.penalty_score,
            'connectedness_score': self.connectedness_score,
            'plausibility_score': self.plausibility_score
        }

    @classmethod
    def from_json(cls, seed_json: Dict, json_graph: JsonGraph):
        temporal_constraints_json = seed_json.get('temporal_constraints', None)
        temporal_constraints = {}
        if temporal_constraints_json is not None:
            for qvar, constraint_json in temporal_constraints_json.items():
                temporal_constraints[qvar] = {
                    key: AidaIncompleteDate.from_json(val) for key, val in constraint_json.items()}

        hypothesis = AidaHypothesis.from_json(
            json_obj=seed_json.get('hypothesis', {}), json_graph=json_graph, weight=0.0)

        return cls(
            json_graph=json_graph,
            core_constraints=seed_json.get('core_constraints', []),
            temporal_constraints=temporal_constraints,
            hypothesis=hypothesis,
            unfilled=set(seed_json.get('unfilled', [])),
            unfillable=set(seed_json.get('unfillable', [])),
            qvar_filler=seed_json.get('qvar_filler', {}),
            entrypoints=seed_json.get('entrypoints', []),
            entrypoint_score=seed_json.get('entrypoint_score', 0.0),
            penalty_score=seed_json.get('penalty_score', 0.0),
            connectedness_score=seed_json.get('connectedness_score', None),
            plausibility_score=seed_json.get('plausibility_score', None)
        )

    # report failed queries of the underlying AidaHypothesis object
    def finalize(self):
        self.hypothesis.add_failed_queries([self.core_constraints[idx] for idx in self.unfillable])
        self.hypothesis.add_qvar_filler(self.qvar_filler)

        return self.hypothesis

    # add typing statements and affiliation statements
    def hypothesis_completion(self):
        self.hypothesis.type_completion()
        self.hypothesis.affiliation_completion()

    # get a tuple of scores for the hypothesis seed, used in rank_seeds in HypothesisSeedManager
    def get_scores(self):
        if self.connectedness_score is None:
            self.connectedness_score = self.hypothesis.get_connectedness_score()

        scores = (self.entrypoint_score,)
        if self.plausibility_score is not None:
            scores += (self.plausibility_score,)
        scores += (self.penalty_score,
                   self.hypothesis.get_connectedness_score())

        return scores

    # extend hypothesis by one statement filling the next fillable core constraint.
    # returns a list of HypothesisSeed objects
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

        elif nfc.failed:
            # this particular constraint was not fillable, and will never be fillable.
            self.unfilled.remove(nfc.constraint_index)
            self.unfillable.add(nfc.constraint_index)
            # adding failed core constraint penalty
            self.penalty_score += self.FAILED_CORE
            return [self]

        else:
            # find statements that match this constraint, and return a list of extension candidates
            # to match this. these candidates have not been run through the filter yet.
            # format: list of tuples (new_hypothesis, stmt_label, variable, filler)
            ext_candidates = self._extend(nfc)

            if len(ext_candidates) == 0:
                # something has gone wrong
                self.unfilled.remove(nfc.constraint_index)
                self.unfillable.add(nfc.constraint_index)
                # adding failed core constraint penalty
                self.penalty_score += self.FAILED_CORE
                return [self]

            new_seeds = []
            for ext_candidate in ext_candidates:
                add_weight = ext_candidate.obj_type_penalty

                if self.filter.validate(ext_candidate.new_hypothesis, ext_candidate.stmt_label):
                    # yes: make a new HypothesisSeed object with this extended hypothesis
                    new_qvar_filler = self.qvar_filler.copy()
                    if ext_candidate.variable and ext_candidate.filler and \
                            self.json_graph.is_ere(ext_candidate.filler):
                        new_qvar_filler[ext_candidate.variable] = ext_candidate.filler
                        if ext_candidate.filler in self.qvar_filler.values():
                            # some other variable has been mapped to the same ERE,
                            # adding duplicate filler penalty
                            add_weight += self.DUPLICATE_FILLER

                    # changes to unfilled, not to unfillable
                    new_unfilled = self.unfilled.difference([nfc.constraint_index])
                    new_unfillable = self.unfillable.copy()

                    if nfc.relaxed_ontology:
                        # adding failed ontology penalty
                        add_weight += self.FAILED_ONTOLOGY

                    new_seeds.append(HypothesisSeed(
                        self.json_graph, self.core_constraints, self.temporal_constraints,
                        hypothesis=ext_candidate.new_hypothesis,
                        qvar_filler=new_qvar_filler,
                        unfilled=new_unfilled,
                        unfillable=new_unfillable,
                        entrypoints=self.entrypoints,
                        entrypoint_score=self.entrypoint_score,
                        penalty_score=self.penalty_score + add_weight))

            if len(new_seeds) == 0:
                # all the fillers were filtered away
                self.unfilled.remove(nfc.constraint_index)
                self.unfillable.add(nfc.constraint_index)
                # all candidate statements filtered away, adding failed core constraint penalty
                self.penalty_score += self.FAILED_CORE
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

    # next fillable constraint from the core constraints list, or None if none fillable
    def _next_fillable_constraint(self) -> Optional[NextFillableConstraint]:
        # iterate over unfilled core constraints to see if we can find one that can be filled
        for constraint_index in self.unfilled:
            subj, pred, obj, _ = self.core_constraints[constraint_index]

            # if either subj or obj is known (is an ERE or has an entry in qvar_filler,
            # then we should be able to fill this constraint now, or it is unfillable
            subj_filler = self._known_core_constraint_entry(subj)
            obj_filler = self._known_core_constraint_entry(obj)

            if subj_filler is not None and obj_filler is not None:
                # new edge between two known variables
                return self._fill_constraint_two_known_eres(
                    constraint_index=constraint_index,
                    subj_ere=subj_filler,
                    pred=pred,
                    obj_ere=obj_filler)

            elif subj_filler is not None:
                # the subject filler is known while the object filler is not
                return self._fill_constraint_one_known_ere(
                    constraint_index=constraint_index,
                    known_ere=subj_filler,
                    known_role='subject',
                    pred=pred,
                    unknown_variable=obj,
                    unknown_role='object'
                )

            elif obj_filler is not None:
                # the object filler is known while the subject filler is not
                return self._fill_constraint_one_known_ere(
                    constraint_index=constraint_index,
                    known_ere=obj_filler,
                    known_role='object',
                    pred=pred,
                    unknown_variable=subj,
                    unknown_role='subject'
                )

            else:
                # both subject filler and object filler are unknown at this point,
                # skip it and see if it can be filled some other time
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
                                       unknown_variable, unknown_role) -> NextFillableConstraint:
        # find statements that could potentially fill the role
        candidates, relaxed = self._statement_candidates(known_ere, pred, known_role)

        if candidates is None:
            # no candidates found at all, constraint is unfillable
            return NextFillableConstraint(constraint_index=constraint_index, failed=True)

        # check if unknown is a constant in the graph, in which case it is not really unknown
        if self._is_string_constant(unknown_variable):
            # which of the statement candidates have the right filler?
            candidates = [s for s in candidates
                          if self.json_graph.node_dict[s][unknown_role] == unknown_variable]
            if len(candidates) == 0:
                return NextFillableConstraint(constraint_index=constraint_index, failed=True)
            else:
                return NextFillableConstraint(
                    constraint_index=constraint_index,
                    failed=False,
                    candidate_stmts=candidates,
                    relaxed_ontology=relaxed,
                    has_variable=False
                )

        else:
            # nope, we have a variable we can fill, any fillers?
            return NextFillableConstraint(
                constraint_index=constraint_index,
                failed=False,
                candidate_stmts=candidates,
                relaxed_ontology=relaxed,
                has_variable=True,
                open_variable=unknown_variable,
                open_role=unknown_role
            )

    # try to fill this constraint from the graph, either strictly or leniently.
    # both sides of this constraint are known EREs
    def _fill_constraint_two_known_eres(self, constraint_index, subj_ere, pred, obj_ere) -> \
            NextFillableConstraint:
        # find statements that could fill the subject role
        possible_candidates, relaxed = self._statement_candidates(subj_ere, pred, 'subject')

        if possible_candidates is None:
            # no candidates found at all, constraint is unfillable
            return NextFillableConstraint(constraint_index=constraint_index, failed=True)

        # we did find candidates. check whether any of the candidates has obj_ere as its object
        candidates = [c for c in possible_candidates if self.json_graph.stmt_object(c) == obj_ere]
        if len(candidates) == 0:
            # constraint is unfillable
            return NextFillableConstraint(constraint_index=constraint_index, failed=True)

        else:
            return NextFillableConstraint(
                constraint_index=constraint_index,
                failed=False,
                candidate_stmts=candidates,
                relaxed_ontology=relaxed,
                has_variable=False
            )

    # find statements that match this constraint, and return a list of ExtensionCandidates:
    def _extend(self, nfc: NextFillableConstraint) -> List[ExtensionCandidate]:

        # did not find any matches to this constraint
        if len(nfc.candidate_stmts) == 0:
            return []

        # this next fillable constraint states a constant string value about a known ERE,
        # or it states a new connection between known EREs. If we do have more than one
        # matching statements. add just the first one, they are identical
        if not nfc.has_variable:
            stmt_label = nfc.candidate_stmts[0]
            if not self.json_graph.has_node(stmt_label):
                logging.warning(f'Statement {stmt_label} returned by _next_fillable_constraint '
                                f'not found in the graph.')
                return []

            else:
                # can this statement be added to the hypothesis without contradiction?
                # extended hypothesis
                return [ExtensionCandidate(
                    new_hypothesis=self.hypothesis.extend(stmt_label, core=True),
                    stmt_label=stmt_label)]

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

    # find statements that match this constraint, and return a list of ExtensionCandidates, and
    # has_temporal_constraint, which is true if there was at least one temporal constraint violated
    def _extend_with_variable(self, nfc: NextFillableConstraint, leeway=0) -> \
            Tuple[List[ExtensionCandidate], bool]:
        # if nfc.open_role is not one of subject / object, we should not be in this method,
        # there must be some error
        if nfc.open_role not in ['subject', 'object']:
            print(f'HypothesisSeed error: unknown open role {nfc.open_role}')
            return [], False

        ext_candidates = []
        has_temporal_constraint = False

        for stmt_label in nfc.candidate_stmts:
            if not self.json_graph.has_node(stmt_label):
                logging.warning(f'Statement {stmt_label} returned by _next_fillable_constraint '
                                f'not found in the graph.')
                continue

            # determine the ERE or value that fills the role that has the variable
            open_filler = self.json_graph.stmt_arg_by_role(stmt_label, nfc.open_role)

            # is this an ERE? if so, we need to check for temporal constraints.
            if self.json_graph.is_ere(open_filler):
                # we also check whether including this statement will violate another constraint.
                # if so, we do not include it
                if self._second_constraint_violated(nfc.open_variable, open_filler,
                                                    nfc.constraint_index):
                    continue

                # is there a problem with a temporal constraint?
                if not temporal_constraint_match(
                        self.json_graph.node_dict[open_filler],
                        self.temporal_constraints.get(nfc.open_variable, None), leeway):
                    # if this filler runs afoul of some temporal constraint, do not use it
                    has_temporal_constraint = True
                    continue

            # extended hypothesis
            new_hypothesis = self.hypothesis.extend(stmt_label, core=True)

            # compute possible violations of objType
            obj_type_penalty = 0.0
            if nfc.open_role == 'object':
                obj_type = self.core_constraints[nfc.constraint_index][-1]
                if obj_type is not None:
                    if obj_type not in self.json_graph.ere_types(open_filler):
                        obj_type_penalty += self.FAILED_OBJ_TYPE

            ext_candidates.append(ExtensionCandidate(
                new_hypothesis=new_hypothesis,
                stmt_label=stmt_label,
                variable=nfc.open_variable,
                filler=open_filler,
                obj_type_penalty=obj_type_penalty))

        return ext_candidates, has_temporal_constraint

    # second constraint violated: given a variable and its filler, see if filling this qvar with
    # this filler will make any constraint that is yet unfilled unfillable
    def _second_constraint_violated(self, variable, filler, except_index):
        for constraint_index in self.unfilled:
            if constraint_index == except_index:
                # this was the constraint we were just going to fill, don't re-check it
                continue

            subj, pred, obj, _ = self.core_constraints[constraint_index]

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

    # is the given string a variable, or should it be viewed as a string constant?
    # use the list of all string constants in the given graph
    def _is_string_constant(self, node_label):
        return node_label in self.json_graph.string_constants
