"""
Author: Katrin Erk, Apr 2019
- Class for rule-based filtering hypotheses for logical consistency
- Filter tests: they take in a hypothesis-under-construction, the statement (which is part of the
    hypothesis) to test, and (optionally) the full hypothesis (if that is available).
    - Some filters only work post-hoc, namely the ones that make use of the full hypothesis.
    - They return False if there is a problem, and True otherwise
    - They assume the hypothesis is error-free except for possibly this statement.
        (This can be achieved by filtering as each new statement is added)

Update: Pengxiang, Aug 2020
- Refactoring and clean-up
- Might need some further clean-up on unifying ad-hoc and post-hoc filters
"""

from collections import deque
from typing import List

from aida_utexas.aif import JsonGraph
from aida_utexas.hypothesis.aida_hypothesis import AidaHypothesis


class AidaHypothesisFilter:
    def __init__(self, json_graph: JsonGraph):
        self.json_graph = json_graph

    def event_attack_attacker_instrument_compatible(self, hypothesis: AidaHypothesis,
                                                    test_stmt: str):
        """
        All attackers in a conflict.attack event need to have one possible affiliation in common,
        also all instruments, and all attackers and instruments
        If there is no known affiliation that is also okay.
        Entities that are possible affiliates of any affiliation relation are counted as their own
        affiliates. For example, Ukraine counts as being affiliated with Ukraine.

        This filter does not use the full hypothesis, so it can be used in hypothesis construction.
        """
        # is stmt an event role of a conflict.attack event, specifically an attacker or instrument?
        if not self.json_graph.is_event_role_stmt(test_stmt):
            return True
        if not self.json_graph.stmt_predicate(test_stmt).startswith('Conflict.Attack'):
            return True
        if not (self.json_graph.stmt_predicate(test_stmt).endswith('Instrument') or
                self.json_graph.stmt_predicate(test_stmt).endswith('Attacker')):
            return True

        # this is an event role of a conflict.attack, so its subject is the event ERE.
        event_label = self.json_graph.stmt_subject(test_stmt)

        # getting argument EREs for attackers and instruments
        attackers = list(hypothesis.event_rel_each_arg_labeled_like(
            event_label, 'Conflict.Attack', 'Attacker'))
        instruments = list(hypothesis.event_rel_each_arg_labeled_like(
            event_label, 'Conflict.Attack', 'Instrument'))

        # if there are multiple attackers but no joint affiliation: problem
        attacker_affl_intersect = self._possible_affiliation_intersect(attackers)
        if attacker_affl_intersect is not None and len(attacker_affl_intersect) == 0:
            return False

        # if there are multiple instrument but no joint affiliation: problem
        instrument_affl_intersect = self._possible_affiliation_intersect(instruments)
        if instrument_affl_intersect is not None and len(instrument_affl_intersect) == 0:
            return False

        # if there is not joint affiliation between attacker(s) and instrument(s): problem
        # noinspection PyUnresolvedReferences
        if attacker_affl_intersect is not None and instrument_affl_intersect is not None and \
                len(attacker_affl_intersect.intersection(instrument_affl_intersect)) == 0:
            return False

        return True

    # helper functions: intersection of possible affiliation labels of EREs.
    # returns None if no known affiliations
    def _possible_affiliation_intersect(self, ere_labels: List[str]):
        joint_affiliations = None

        for ere_label in ere_labels:
            affiliations = set(self.json_graph.ere_affiliations(ere_label))
            if self.json_graph.is_ere_affiliation(ere_label):
                affiliations.add(ere_label)
            if len(affiliations) > 0:
                if joint_affiliations is None:
                    joint_affiliations = affiliations
                else:
                    joint_affiliations.intersection_update(affiliations)

        return joint_affiliations

    def event_attack_all_roles_different(self, hypothesis: AidaHypothesis, test_stmt: str):
        """
        All roles of an attack event need to be filled by different fillers (no attacking yourself)
        This filter does not use the full hypothesis, so it can be used in hypothesis construction.
        """
        if not self.json_graph.is_event_role_stmt(test_stmt):
            return True
        if not self.json_graph.stmt_predicate(test_stmt).startswith("Conflict.Attack"):
            return True

        event_label = self.json_graph.stmt_subject(test_stmt)
        event_arg_label = self.json_graph.stmt_object(test_stmt)

        if any(arg_label == event_arg_label for stmt_label, _, arg_label
               in hypothesis.event_rel_each_arg_stmt(event_label) if stmt_label != test_stmt):
            return False

        return True

    def single_type_per_event_rel(self, hypothesis: AidaHypothesis, test_stmt: str,
                                  full_hypothesis: AidaHypothesis):
        """
        Don't have multiple types on an event or relation.
        This filter takes the full hypothesis into account and hence only works post-hoc.
        """
        # potential problem only if this is a type statement
        if not self.json_graph.is_type_stmt(test_stmt):
            return True

        ere_label = self.json_graph.stmt_subject(test_stmt)

        # no problem if we have an entity
        if self.json_graph.is_entity(ere_label):
            return True

        # we have an event or relation: now check whether this ere has another type
        types = list(hypothesis.ere_types(ere_label))
        if len(types) > 1:
            return False

        # this is an only type: check to see whether it coincides with any roles of this event or
        # relation IN THE FULL HYPOTHESIS (this is the part that only works post-hoc)
        role_labels = [self.json_graph.shorten_label(pred_label) for pred_label, arg_label
                       in full_hypothesis.event_rel_each_arg(ere_label)]
        type_label = self.json_graph.shorten_label(self.json_graph.stmt_object(test_stmt))
        if not any(role_label.startswith(type_label) for role_label in role_labels):
            # no, this type label does not coincide with any role label
            return False

        return True

    def relations_need_two_args(self, hypothesis: AidaHypothesis, test_stmt: str,
                                full_hypothesis: AidaHypothesis):
        """
        Don't have relations with only one argument.
        This filter takes the full hypothesis into account and hence only works post-hoc.
        """
        # is this an argument of a relation
        if not self.json_graph.is_relation_role_stmt(test_stmt):
            return True

        rel_label = self.json_graph.stmt_subject(test_stmt)

        # check if this relation ERE has more than one argument IN THE FULL HYPOTHESIS
        # (this is the part that only works post-hoc)
        if len(list(full_hypothesis.event_rel_each_arg(rel_label))) > 1:
            return True

        # this is the only argument of this relation. don't add it
        return False

    def events_need_two_args(self, hypothesis: AidaHypothesis, test_stmt: str,
                             full_hypothesis: AidaHypothesis):
        """
        Don't have events with only one argument, except when they are core EREs (that is,
        adjacent to one of the core statements).
        This filter takes the full hypothesis into account and hence only works post-hoc.
        """
        # is this an argument of an event
        if not self.json_graph.is_event_role_stmt(test_stmt):
            return True

        event_label = self.json_graph.stmt_subject(test_stmt)
        arg_label = self.json_graph.stmt_object(test_stmt)

        # if the argument is a core ERE, it's fine, keep the one-argument event
        if arg_label in hypothesis.core_eres():
            return True

        # check if this event ERE has more than one argument IN THE FULL HYPOTHESIS
        # (this is the part that only works post-hoc)
        if len(list(full_hypothesis.event_rel_each_arg(event_label))) > 1:
            return True

        # this is the only argument of this event. don't add it
        return False

    # main checking function: check one single statement, which is part of the hypothesis.
    # assumption: this statement is the only potentially broken statement in the hypothesis
    def validate(self, hypothesis: AidaHypothesis, test_stmt: str,
                 full_hypothesis: AidaHypothesis = None):
        for test_func in [self.event_attack_attacker_instrument_compatible,
                          self.event_attack_all_roles_different]:
            if not test_func(hypothesis, test_stmt):
                return False

        # post-hoc filtering, requires full hypothesis
        if full_hypothesis is not None:
            for test_func in [self.single_type_per_event_rel,
                              self.relations_need_two_args,
                              self.events_need_two_args]:
                if not test_func(hypothesis, test_stmt, full_hypothesis):
                    return False

        return True

    # other main function:
    # post-hoc, remove statements from the hypothesis that shouldn't be there.
    # do this by starting a new hypothesis and re-inserting statements there by statement weight,
    # using the validate function
    def filtered(self, hypothesis: AidaHypothesis):
        # new hypothesis: start with the core statements, then add in things one at a time.
        new_hypothesis = AidaHypothesis(
            json_graph=self.json_graph,
            stmts=hypothesis.core_stmts.copy(),
            core_stmts=hypothesis.core_stmts.copy(),
            stmt_weights={
                stmt: stmt_weight for stmt, stmt_weight in hypothesis.stmt_weights.items()
                if stmt in hypothesis.core_stmts
            },
            weight=hypothesis.weight,
            questionIDs=hypothesis.questionIDs
        ) # questionIDs added

        new_hypothesis.add_failed_queries(hypothesis.failed_queries)
        new_hypothesis.add_qvar_filler(hypothesis.qvar_filler)

        new_hypothesis_eres = set(new_hypothesis.eres())

        # all non-core statements are candidates, sorted by their weights, highest first
        candidates = [stmt for stmt in hypothesis.stmts if stmt not in hypothesis.core_stmts]
        candidates.sort(key=lambda s: hypothesis.stmt_weights[s], reverse=True)
        candidates = deque(candidates)

        # candidates are set aside if they currently don't connect to any ERE in the new hypothesis
        candidates_set_aside = deque()

        while len(candidates) > 0:
            # any set-aside candidates that turn out to be connected to the hypothesis after all?
            resurrected_stmts = []
            for stmt in candidates_set_aside:
                if any(ere in new_hypothesis_eres for ere in self.json_graph.stmt_args(stmt)):
                    # yes, now check whether this candidate should be inserted
                    resurrected_stmts.append(stmt)
                    new_hypothesis, new_eres = self._test_and_insert_candidate(
                        stmt, new_hypothesis, hypothesis)
                    new_hypothesis_eres.update(new_eres)

            for stmt in resurrected_stmts:
                candidates_set_aside.remove(stmt)

            # now test the next non-set-aside candidate
            stmt = candidates.popleft()
            # does it need to be set aside?
            if not any(ere in new_hypothesis_eres for ere in self.json_graph.stmt_args(stmt)):
                candidates_set_aside.append(stmt)
            else:
                # no, we can test this one now.
                new_hypothesis, new_eres = self._test_and_insert_candidate(
                    stmt, prev_hypothesis=new_hypothesis, full_hypothesis=hypothesis)
                new_hypothesis_eres.update(new_eres)

        # now no candidates left in the candidate set, but maybe something from the set-aside
        # candidate list has become connected to the core by the last candidate to be added
        for stmt in candidates_set_aside:
            if any(ere in new_hypothesis_eres for ere in self.json_graph.stmt_args(stmt)):
                # yes, check now whether this candidate should be inserted
                new_hypothesis, new_eres = self._test_and_insert_candidate(
                    stmt, prev_hypothesis=new_hypothesis, full_hypothesis=hypothesis)
                new_hypothesis_eres.update(new_eres)

        return new_hypothesis

    def _test_and_insert_candidate(self, stmt, prev_hypothesis, full_hypothesis):
        test_hypothesis = prev_hypothesis.extend(stmt, weight=full_hypothesis.stmt_weights[stmt])

        if self.validate(test_hypothesis, stmt, full_hypothesis):
            # yes, statement is fine. add the statement's EREs to the new set of EREs
            new_eres = [ere_label for ere_label in self.json_graph.stmt_args(stmt)
                        if self.json_graph.is_ere(ere_label)]
            return test_hypothesis, new_eres
        else:
            # don't add stmt after all
            return prev_hypothesis, []
