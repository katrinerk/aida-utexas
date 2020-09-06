"""
Author: Katrin Erk, Mar 2019
- Rule-based expansion of hypotheses

Update: Pengxiang Cheng, Aug 2020
- Use new JsonGraph API
"""

from collections import defaultdict
from typing import Dict, List

from aida_utexas.aif import JsonGraph
from aida_utexas.seeds.aidahypothesis import AidaHypothesisCollection, AidaHypothesis


# class that manages cluster expansion
class ClusterExpansion:
    # initialize with a JsonGraph and an AidaHypothesisCollection
    def __init__(self, json_graph: JsonGraph, hypothesis_collection: AidaHypothesisCollection):
        self.json_graph = json_graph
        self.hypothesis_collection = hypothesis_collection

    def to_json(self):
        return self.hypothesis_collection.to_json()

    # make a list of strings for human-readable output
    def to_str(self, roles_ontology: Dict):
        return self.hypothesis_collection.to_str(roles_ontology)

    # For each ERE, add *all* typing statements.
    # This is done using add_stmt and not extend (which calls filter)
    # because all type statements are currently thought to be compatible.
    # This code is not used anymore because the eval document wants to see only those event types
    # that match role labels
    def type_completion_all(self):
        for hypothesis in self.hypothesis_collection.hypotheses:
            for ere_label in hypothesis.eres():
                for stmt_label in self.json_graph.each_ere_adjacent_stmt(
                        ere_label, predicate='type', ere_role='subject'):
                    hypothesis.add_stmt(stmt_label)
            # add weights for the types we have been adding
            hypothesis.set_type_stmt_weights()

    # For each ERE, add typing statements.
    # For entities, add all typing statements.
    # For events and relations, include only those types that match at least one role statement.
    # This will still add multiple event/relation types in case there are roles matching different
    # event/relation types.
    # It's up to the final filter to remove those if desired.
    def type_completion(self):
        for hypothesis in self.hypothesis_collection.hypotheses:
            # map each ERE to adjacent type statements
            ere_to_type_stmts = defaultdict(list)
            for ere_label in hypothesis.eres():
                for stmt_label in self.json_graph.each_ere_adjacent_stmt(
                        ere_label, predicate='type', ere_role='subject'):
                    ere_to_type_stmts[ere_label].append(stmt_label)

            # for each ERE in the hypothesis, add types matching a role statement
            for ere, type_stmts in ere_to_type_stmts.items():
                # entity: add all types.
                if self.json_graph.is_entity(ere):
                    for stmt in type_stmts:
                        hypothesis.add_stmt(stmt)
                else:
                    # event or relation: only add types that match a role included in the hypothesis
                    role_labels = [self.json_graph.shorten_label(pred_label)
                                   for pred_label, _ in hypothesis.event_rel_each_arg(ere)]

                    for type_stmt in type_stmts:
                        type_label = self.json_graph.stmt_object(type_stmt)
                        if type_label is None:
                            # weird type statement, skip
                            continue

                        type_label = self.json_graph.shorten_label(type_label)
                        if any(role_label.startswith(type_label) for role_label in role_labels):
                            # this is a type that is used in an outgoing edge of this ERE
                            hypothesis.add_stmt(type_stmt)

    # for each entity, add all affiliation statements.
    def affiliation_completion(self):
        for hypothesis in self.hypothesis_collection.hypotheses:
            # iterate over EREs
            for ere in hypothesis.eres():
                # collect all affiliation labels for this ERE
                possible_affiliations = set(self.json_graph.ere_affiliation_triples(ere))
                # possibly multiple affiliation statements, but all point to the same affiliation.
                # add one of them.
                if len(possible_affiliations) == 1:
                    for stmt1, _, stmt2 in self.json_graph.ere_affiliation_triples(ere):
                        hypothesis.add_stmt(stmt1)
                        hypothesis.add_stmt(stmt2)
                        break

    def hypotheses(self) -> List[AidaHypothesis]:
        return self.hypothesis_collection.hypotheses
