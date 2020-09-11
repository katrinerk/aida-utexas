"""
Author: Katrin Erk, Apr 2019
- Simple class for keeping an AIDA hypothesis.
- As a data structure, a hypothesis is simply a list of statements.
- We additionally track core statements so they can be visualized separately if needed.
- The class also provides access functions to determine EREs that are mentioned in the hypothesis,
    as well as their types, arguments, etc.
- All this is computed dynamically and not kept in the data structure.

Update: Pengxiang Cheng, Aug 2020
- Use new JsonGraph API
- Refactor and clean-up

Update: Pengxiang Cheng, Sep 2019
- Merge methods in ClusterExpansion
"""

from collections import defaultdict
from typing import Dict, Iterable, List

from aida_utexas.aif import JsonGraph


class AidaHypothesis:
    def __init__(self, json_graph: JsonGraph, stmts: Iterable = None, stmt_weights: Dict = None,
                 core_stmts: Iterable = None, weight: float = 0.0):
        self.json_graph = json_graph

        # list of all statements in the hypothesis
        self.stmts = set(stmts) if stmts is not None else set()

        # mapping from statement labels to their weights
        self.stmt_weights = stmt_weights if stmt_weights is not None else {}

        # list of core statements in the hypothesis
        self.core_stmts = set(core_stmts) if core_stmts is not None else set()

        # hypothesis weight
        self.weight = weight

        # failed queries are added from outside, as they are needed in the json object
        self.failed_queries = []

        # query variables and fillers, for quick access to the core answer that this query gives
        self.qvar_filler = {}

        # default weight for non-core statements
        self.default_stmt_weight = -100.0

    # extending a hypothesis: adding a statement in place
    def add_stmt(self, stmt_label: str, core: bool = False, weight: float = None):
        if stmt_label in self.stmts:
            return

        self.stmts.add(stmt_label)

        if core:
            self.core_stmts.add(stmt_label)

        # if weight is provided, use it
        if weight is not None:
            self.stmt_weights[stmt_label] = weight
        # otherwise if it is a core statement, use weight 0.0
        elif core:
            self.stmt_weights[stmt_label] = 0.0
        # otherwise, use the default weight for non-core statements
        else:
            self.stmt_weights[stmt_label] = self.default_stmt_weight

    # extending a hypothesis: making a new hypothesis and adding the statement there
    def extend(self, stmt_label: str, core: bool = False, weight: float = None):
        if stmt_label in self.stmts:
            return self

        new_hypothesis = self.copy()
        new_hypothesis.add_stmt(stmt_label, core, weight)
        return new_hypothesis

    def copy(self):
        new_hypothesis = AidaHypothesis(
            json_graph=self.json_graph,
            stmts=self.stmts.copy(),
            core_stmts=self.core_stmts.copy(),
            stmt_weights=self.stmt_weights.copy(),
            weight=self.weight
        )
        new_hypothesis.add_failed_queries(self.failed_queries)
        return new_hypothesis

    # connectedness score of the hypothesis: average number of adjacent statements of all EREs
    def get_connectedness_score(self):
        total_score = 0
        num_eres = 0

        for ere_label in self.eres():
            # count the number of statements adjacent to the ERE
            total_score += len(list(self.json_graph.each_ere_adjacent_stmt(ere_label)))
            # update the ERE count
            num_eres += 1

        return total_score / num_eres

    # set weights of type statements in this hypothesis:
    # for single type, or type of an event/relation used in an event/relation argument: set it to
    # the weight of maximum neighbor.
    # otherwise, set it to the low default weight.
    def set_type_stmt_weights(self):

        # map each ERE to adjacent type statements
        ere_to_type_stmts = defaultdict(list)
        for stmt in self.stmts:
            if self.json_graph.is_type_stmt(stmt):
                ere_label = self.json_graph.stmt_subject(stmt)
                if ere_label is not None:
                    ere_to_type_stmts[ere_label].append(stmt)

        # for each ERE adjacent to some type statements
        for ere, type_stmts in ere_to_type_stmts.items():
            # get the weight of the maximum-weight edge adjacent to the ERE
            ere_max_weight = max(self.stmt_weights.get(stmt, self.default_stmt_weight)
                                 for stmt in self.json_graph.each_ere_adjacent_stmt(ere))

            # only one type statement for the ERE: set the type statement weight to ere_max_weight
            if len(type_stmts) == 1:
                self.stmt_weights[type_stmts[0]] = ere_max_weight

            else:
                # multiple type statements for ere
                # what outgoing event/relation edges does it have?
                edge_role_labels = [self.json_graph.shorten_label(pred_label)
                                    for pred_label, _ in self.event_rel_each_arg(ere)]

                for type_stmt in type_stmts:
                    type_label = self.json_graph.stmt_object(type_stmt)
                    if type_label is None:
                        self.stmt_weights[type_label] = self.default_stmt_weight
                        continue

                    type_label = self.json_graph.shorten_label(type_label)
                    if any(role_label.startswith(type_label) for role_label in edge_role_labels):
                        # this is a type that is used in an outgoing edge of this ERE
                        self.stmt_weights[type_stmt] = ere_max_weight
                    else:
                        # no reason not to give a low default weight to this edge
                        self.stmt_weights[type_stmt] = self.default_stmt_weight

    # For each ERE, add *all* typing statements.
    # This code is not used anymore because the eval document wants to see only those event types
    # that match role labels
    def type_completion_all(self):
        for ere_label in self.eres():
            for stmt_label in self.json_graph.each_ere_adjacent_stmt(
                    ere_label, predicate='type', ere_role='subject'):
                self.add_stmt(stmt_label)
        # add weights for the types we have been adding
        self.set_type_stmt_weights()

    # For each ERE, add typing statements.
    # For entities, add all typing statements.
    # For events and relations, include only those types that match at least one role statement.
    # This will still add multiple event/relation types in case there are roles matching different
    # event/relation types.
    # It's up to the final filter to remove those if desired.
    def type_completion(self):
        # map each ERE to adjacent type statements
        ere_to_type_stmts = defaultdict(list)
        for ere_label in self.eres():
            for stmt_label in self.json_graph.each_ere_adjacent_stmt(
                    ere_label, predicate='type', ere_role='subject'):
                ere_to_type_stmts[ere_label].append(stmt_label)

        # for each ERE in the hypothesis, add types matching a role statement
        for ere, type_stmts in ere_to_type_stmts.items():
            # entity: add all types.
            if self.json_graph.is_entity(ere):
                for stmt in type_stmts:
                    self.add_stmt(stmt)
            else:
                # event or relation: only add types that match a role included in the hypothesis
                role_labels = [self.json_graph.shorten_label(pred_label)
                               for pred_label, _ in self.event_rel_each_arg(ere)]

                for type_stmt in type_stmts:
                    type_label = self.json_graph.stmt_object(type_stmt)
                    if type_label is None:
                        # weird type statement, skip
                        continue

                    type_label = self.json_graph.shorten_label(type_label)
                    if any(role_label.startswith(type_label) for role_label in role_labels):
                        # this is a type that is used in an outgoing edge of this ERE
                        self.add_stmt(type_stmt)

    # for each entity, add all affiliation statements.
    def affiliation_completion(self):
        for ere in self.eres():
            # collect all affiliation labels for this ERE
            possible_affiliations = set(self.json_graph.ere_affiliation_triples(ere))
            # possibly multiple affiliation statements, but all point to the same affiliation.
            # add one of them.
            if len(possible_affiliations) == 1:
                for stmt1, _, stmt2 in self.json_graph.ere_affiliation_triples(ere):
                    self.add_stmt(stmt1)
                    self.add_stmt(stmt2)
                    break

    # update hypothesis weight
    def update_weight(self, added_weight):
        self.weight += added_weight

    def add_failed_queries(self, failed_queries):
        self.failed_queries = failed_queries

    def add_qvar_filler(self, qvar_filler):
        self.qvar_filler = qvar_filler

    # json output of the hypothesis
    def to_json(self):
        stmt_list = list(self.stmts)
        stmt_weight_list = [self.stmt_weights.get(s, self.default_stmt_weight) for s in stmt_list]

        return {
            'statements': stmt_list,
            'statementWeights': stmt_weight_list,
            'failedQueries': self.failed_queries,
            'queryStatements': list(self.core_stmts)
        }

    @classmethod
    def from_json(cls, json_obj: Dict, json_graph: JsonGraph, weight: float = 0.0):
        hypothesis = cls(
            json_graph=json_graph,
            stmts=json_obj['statements'],
            core_stmts=json_obj['queryStatements'],
            stmt_weights=dict(zip(json_obj['statements'], json_obj['statementWeights'])),
            weight=weight)

        hypothesis.add_failed_queries(json_obj['failedQueries'])
        return hypothesis

    # human-readable output for an ERE
    def ere_to_str(self, ere_label, roles_ontology: Dict):
        if self.json_graph.is_event(ere_label) or self.json_graph.is_relation(ere_label):
            return self.event_rel_to_str(ere_label, roles_ontology)
        elif self.json_graph.is_entity(ere_label):
            return self.entity_to_str(ere_label)
        else:
            return ''

    # human-readable output for an entity
    def entity_to_str(self, ere_label):
        result = self.entity_best_name(ere_label)
        if result == '[unknown]':
            for type_label in self.ere_types(ere_label):
                result = type_label
                break
        return result

    # human-readable output for an event or relation
    def event_rel_to_str(self, ere_label, roles_ontology: Dict):
        if not (self.json_graph.is_event(ere_label) or self.json_graph.is_relation(ere_label)):
            return ''

        event_rel_type = None
        for type_label in self.ere_types(ere_label):
            event_rel_type = type_label
            break

        event_rel_roles = defaultdict(set)
        for pred_label, arg_label in self.event_rel_each_arg(ere_label):
            event_rel_roles[pred_label].add(arg_label)

        if not event_rel_roles:
            return ''

        if event_rel_type is None:
            event_rel_type = list(event_rel_roles.keys())[0].rsplit('_', maxsplit=1)[0]

        result = event_rel_type
        for role_label in roles_ontology[event_rel_type].values():
            result += '\n    ' + role_label + ': '
            pred_label = event_rel_type + '_' + role_label
            if pred_label in event_rel_roles:
                result += ', '.join(self.entity_to_str(arg_label)
                                    for arg_label in event_rel_roles[pred_label])

        return result

    # human-readable output for whole hypothesis
    def to_str(self, roles_ontology: Dict):
        result = ''

        core_eres = self.core_eres()

        # start with core EREs
        for ere_label in core_eres:
            if self.json_graph.is_event(ere_label) or self.json_graph.is_relation(ere_label):
                ere_str = self.ere_to_str(ere_label, roles_ontology)
                if ere_str != '':
                    result += ere_str + '\n\n'

        # make output for each non-core event in the hypothesis
        for ere_label in self.eres():
            if ere_label not in core_eres and self.json_graph.is_event(ere_label):
                ere_str = self.ere_to_str(ere_label, roles_ontology)
                if ere_str != '':
                    result += ere_str + '\n\n'

        # make output for each non-core relation in the hypothesis
        for ere_label in self.eres():
            if ere_label not in core_eres and self.json_graph.is_relation(ere_label):
                ere_str = self.ere_to_str(ere_label, roles_ontology)
                if ere_str != '':
                    result += ere_str + '\n\n'

        return result

    # helper functions

    # list of EREs adjacent to the statements in this hypothesis
    def eres(self):
        return list(set(node_label for stmt_label in self.stmts
                        for node_label in self.json_graph.stmt_args(stmt_label)
                        if self.json_graph.is_ere(node_label)))

    # list of EREs adjacent to core statements of this hypothesis
    def core_eres(self):
        return list(set(node_label for stmt_label in self.core_stmts
                        for node_label in self.json_graph.stmt_args(stmt_label)
                        if self.json_graph.is_ere(node_label)))

    # list of EREs adjacent to a statement
    def eres_of_stmt(self, stmt_label):
        if stmt_label not in self.stmts:
            return []
        else:
            return list(set(node_label for node_label in self.json_graph.stmt_args(stmt_label)
                            if self.json_graph.is_ere(node_label)))

    # iterate over arguments of an event or relation in this hypothesis
    # yield tuples of (statement label, predicate label, ERE label)
    def event_rel_each_arg_stmt(self, ere_label):
        if not (self.json_graph.is_event(ere_label) or self.json_graph.is_relation(ere_label)):
            return

        for stmt_label in self.json_graph.each_ere_adjacent_stmt(ere_label, ere_role='subject'):
            stmt_obj = self.json_graph.stmt_object(stmt_label)
            if self.json_graph.is_ere(stmt_obj):
                yield stmt_label, self.json_graph.stmt_predicate(stmt_label), stmt_obj

    # iterate over arguments of an event or relation with pred_label as the predicate
    # yield pairs of (statement label, ERE label)
    def event_rel_each_arg_stmt_labeled(self, ere_label, pred_label):
        for stmt_label, stmt_pred, stmt_obj in self.event_rel_each_arg_stmt(ere_label):
            if stmt_pred == pred_label:
                yield stmt_label, stmt_obj

    # iterate over arguments of an event or relation whose predicate starts with class_label and
    # ends with role_label
    # yield pairs of (statement label, ERE label)
    def event_rel_each_arg_stmt_labeled_like(self, ere_label, class_label, role_label):
        for stmt_label, stmt_pred, stmt_obj in self.event_rel_each_arg_stmt(ere_label):
            if stmt_pred.startswith(class_label) and stmt_pred.endswith(role_label):
                yield stmt_label, stmt_obj

    # iterate over arguments of an event or relation in this hypothesis
    # yield pairs of (predicate label, ERE label)
    def event_rel_each_arg(self, ere_label):
        for _, stmt_pred, stmt_obj in self.event_rel_each_arg_stmt(ere_label):
            yield stmt_pred, stmt_obj

    # return each argument of the event or relation that has pred_label as the predicate
    def event_rel_each_arg_labeled(self, ere_label, pred_label):
        for stmt_pred, stmt_obj in self.event_rel_each_arg(ere_label):
            if stmt_pred == pred_label:
                yield stmt_obj

    # return each argument of the event or relation whose predicate starts with class_label and
    # ends with role_label
    def event_rel_each_arg_labeled_like(self, ere_label, class_label, role_label):
        for stmt_pred, stmt_obj in self.event_rel_each_arg(ere_label):
            if stmt_pred.startswith(class_label) and stmt_pred.endswith(role_label):
                yield stmt_obj

    # types of an ERE node in this hypothesis
    def ere_types(self, ere_label):
        if not self.json_graph.is_ere(ere_label):
            return
        for stmt_label in self.json_graph.each_ere_adjacent_stmt(ere_label, 'type', 'subject'):
            if stmt_label in self.stmts:
                yield self.json_graph.shorten_label(self.json_graph.stmt_object(stmt_label))

    # affiliations of an ERE in this hypothesis
    def ere_affiliations(self, ere_label):
        for stmt1, _, stmt2 in self.json_graph.ere_affiliation_triples(ere_label):
            if stmt1 in self.stmts and stmt2 in self.stmts:
                yield self.json_graph.stmt_object(stmt2)

    # node type: Entity, Event, Relation, Statement
    def node_type(self, node_label):
        if self.json_graph.has_node(node_label):
            return self.json_graph.node_dict[node_label].type
        else:
            return None

    # "best" name of an entity
    def entity_best_name(self, ere_label):
        names = self.json_graph.ere_names(ere_label)
        if names is None or names == []:
            return "[unknown]"
        english_names = self.json_graph.english_names(names)
        if len(english_names) > 0:
            return min(english_names, key=lambda n: len(n))
        else:
            return min(names, key=lambda n: len(n))


# collection of hypotheses, after initial hypothesis seed generation has been done
class AidaHypothesisCollection:
    def __init__(self, hypotheses: List[AidaHypothesis]):
        self.hypotheses = hypotheses

    def __iter__(self):
        for hypothesis in self.hypotheses:
            yield hypothesis

    def add(self, hypothesis):
        self.hypotheses.append(hypothesis)

    # make a list of strings for human-readable output
    def to_str(self, roles_ontology: Dict):
        return [hyp.to_str(roles_ontology) for hyp in self.hypotheses]

    def to_json(self):
        return {
            'probs': [hyp.weight for hyp in self.hypotheses],
            'support': [hyp.to_json() for hyp in self.hypotheses]
        }

    def expand(self):
        for hypothesis in self.hypotheses:
            hypothesis.type_completion()
            hypothesis.affiliation_completion()

    @classmethod
    def from_json(cls, json_obj: Dict, json_graph: JsonGraph):
        return cls([AidaHypothesis.from_json(hyp_json_obj, json_graph, hyp_weight)
                    for hyp_json_obj, hyp_weight in zip(json_obj['support'], json_obj['probs'])])
