"""
Author: Pengxiang Cheng, Aug 2020
- Adapted from the legacy JsonInterface and AidaJson classes by Katrin.
- Json representation for an AIDA graph, with methods to reason over it.

Updated Katrin Erk Nov 2020, Jan 2021
- adapting to COVID domain with claim frames
"""

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List

from tqdm import tqdm

from aida_utexas.aif.aida_graph import AidaGraph

APORA = 'GeneralAffiliation.ArtifactPoliticalOrganizationReligiousAffiliation'
MORE = 'GeneralAffiliation.MemberOriginReligionEthnicity'
OPRA = 'GeneralAffiliation.OrganizationPoliticalReligiousAffiliation'
SPONSOR = 'GeneralAffiliation.Sponsorship'

# role labels of affiliates in an affiliation relation
affiliate_role_labels = [
    APORA + '_Artifact',
    APORA + '.ControlTerritory_Territory',
    APORA + '.NationalityCitizen_Artifact',
    APORA + '.OwnershipPossession_Artifact',
    MORE + '_Person',
    MORE + '.Ethnicity_Person',
    MORE + '.NationalityCitizen_Citizen',
    OPRA + '_Organization',
    OPRA + '.NationalityCitizen_Organization',
    'GeneralAffiliation.OrganizationWebsite.OrganizationWebsite_Website',
    SPONSOR + '_ActorOrEvent',
    SPONSOR + '.AdvisePlanOrganize_ActorOrEvent',
    SPONSOR + '.Affiliated_ActorOrEvent',
    SPONSOR + '.HelpSupport_ActorOrEvent',
    'OrganizationAffiliation.EmploymentMembership_EmployeeMember',
    'OrganizationAffiliation.EmploymentMembership.Employment_Employee',
    'OrganizationAffiliation.EmploymentMembership.Membership_Member',
    'OrganizationAffiliation.Founder.Founder_Founder',
    'OrganizationAffiliation.Leadership_Leader',
    'OrganizationAffiliation.Leadership.Government_Leader',
    'OrganizationAffiliation.Leadership.HeadOfState_Leader',
    'OrganizationAffiliation.Leadership.MilitaryPolice_Leader'
]

# role labels of affiliations in an affiliation relation
affiliation_role_labels = [
    APORA + '_EntityOrFiller',
    APORA + '.ControlTerritory_Controller',
    APORA + '.NationalityCitizen_Nationality',
    APORA + '.OwnershipPossession_Owner',
    MORE + '_EntityOrFiller',
    MORE + '.Ethnicity_Ethnicity',
    MORE + '.NationalityCitizen_Nationality',
    OPRA + '_EntityOrFiller',
    OPRA + '.NationalityCitizen_Nationality',
    'GeneralAffiliation.OrganizationWebsite.OrganizationWebsite_Organization',
    SPONSOR + '_Sponsor',
    SPONSOR + '.AdvisePlanOrganize_Sponsor',
    SPONSOR + '.Affiliated_Sponsor',
    SPONSOR + '.HelpSupport_Sponsor',
    'OrganizationAffiliation.EmploymentMembership_PlaceOfEmploymentMembership',
    'OrganizationAffiliation.EmploymentMembership.Employment_PlaceOfEmployment',
    'OrganizationAffiliation.EmploymentMembership.Membership_PlaceOfMembership',
    'OrganizationAffiliation.Founder.Founder_Organization',
    'OrganizationAffiliation.Leadership_Organization',
    'OrganizationAffiliation.Leadership.Government_GovernmentBodyOrGPE',
    'OrganizationAffiliation.Leadership.HeadOfState_Country',
    'OrganizationAffiliation.Leadership.MilitaryPolice_MilitaryPoliceORG'
]


@dataclass
class ERENode:
    type: str
    index: int
    adjacent: List[str] = field(default_factory=list)
    name: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    ldcTime: List[Dict] = field(default_factory=list)
    kblinks: List[str] = field(default_factory=list)


@dataclass
class StatementNode:
    type: str
    index: int
    subject: str = None
    predicate: str = None
    object: str = None
    attributes: List[str] = field(default_factory=list)
    conf: float = None


@dataclass
class SameAsClusterNode:
    type: str
    prototype: str = None
    handle: str = None


@dataclass
class ClusterMembershipNode:
    type: str
    index: int
    cluster: str = None
    clusterMember: str = None
    conf: float = None

# NEW COVID DOMAIN: claim component
@dataclass
class ClaimComponentNode:
    type: str
    index: int
    name: str
    identity: str
    component_ke: str
    component_provenance: str
    component_type: List[str] = field(default_factory = list)
    
    
# NEW COVID DOMAIN: claim
@dataclass
class ClaimNode:
    type: str
    index: int
    claim_id: str
    query_id: str
    doc_id: str
    topic: str
    subtopic: str
    claim_template: str
    text: str
    sentiment: str
    epistemic: str
    importance: float = None
    xvar: List[str] = field(default_factory=list)
    claimer: List[str] = field(default_factory=list)
    claim_semantics: List[str] = field(default_factory=list)
    associated_kes: List[str] = field(default_factory=list)
    identical_claims: List[str] = field(default_factory=list)
    related_claims: List[str] = field(default_factory=list)
    supporting_claims: List[str] = field(default_factory=list)
    refuting_claims: List[str] = field(default_factory=list)
    date_time: List[Dict] = field(default_factory=list)

    
# UPDATED LE Jan 2022
node_cls_by_type = {
    'Entity': ERENode, 'Relation': ERENode, 'Event': ERENode, 'Statement': StatementNode,
    'SameAsCluster': SameAsClusterNode, 'ClusterMembership': ClusterMembershipNode,
    'Claim' : ClaimNode, 'ClaimComponent' : ClaimComponentNode
}


class JsonGraph:
    def __init__(self):
        self.node_dict = {}

        self.eres = []
        self.statements = []
        self.claims = [ ]

        self.string_constants = None

    def build_graph(self, aida_graph: AidaGraph):
        logging.info('Building JSON graph from the AIDA graph ...')

        self.transform_graph(aida_graph)
        self.validate_graph()

        self.string_constants = list(self.each_string_constant_of_graph())

        logging.info('Done.')

    def transform_graph(self, aida_graph: AidaGraph):
        logging.info('Transforming the AIDA graph ...')

        ere_counter = 0
        stmt_counter = 0
        coref_counter = 0
        # NEW COVID DOMAIN: count claims
        claim_counter = 0

        for node in tqdm(aida_graph.nodes()):
            node_label = node.name
            
            # formatting for EREs
            if node.is_ere():
                if node.is_entity():
                    node_type = 'Entity'
                    # self.entities.append(str(node_label))
                elif node.is_relation():
                    node_type = 'Relation'
                    # self.relations.append(str(node_label))
                else:
                    node_type = 'Event'
                    # self.events.append(str(node_label))

                adjacent_stmts = list(map(str, aida_graph.adjacent_stmts_of_ere(node_label)))
                
                attributes = list(iter(node.get('attributes', shorten=True)))

                # retrieve KB links
                kblinks = list(link for link, confidence in aida_graph.kb_links_of(node_label))
                
                # Retrieve sentence and mention_string
                sentence, mention_string = aida_graph.get_sent_mention_str(node_label)
                node_hasName = list(aida_graph.names_of_ere(node_label))
                if not node_hasName:
                    node_hasName = ['']

                self.node_dict[str(node_label)] = ERENode(
                    type=node_type,
                    index=ere_counter,
                    adjacent=adjacent_stmts,
                    name=node_hasName + [mention_string],
                    attributes = attributes,
                    kblinks=kblinks,
                    ldcTime=list(aida_graph.times_associated_with(node_label))
                ) # Update: Removed 'sentence' in 'name' field and only added 'mention_string' (as we did not observe much effect from using sentence)


                self.eres.append(str(node_label))
                ere_counter += 1

            # formatting for statements
            elif node.is_statement():
                subj = next(iter(node.get('subject', shorten=False)), None)
                pred = next(iter(node.get('predicate', shorten=True)), None)
                obj = next(iter(node.get('object', shorten=False)), None)

                conf_levels = aida_graph.confidence_of(node_label)

                attributes = list(iter(node.get('attributes', shorten=True)))

                self.node_dict[str(node_label)] = StatementNode(
                    type='Statement',
                    index=stmt_counter,
                    subject=str(subj) if subj else None,
                    predicate=str(pred) if pred else None,
                    object=str(obj) if obj else None,
                    attributes = attributes,
                    conf=max(conf_levels) if conf_levels else None
                )

                self.statements.append(str(node_label))
                stmt_counter += 1

            # formatting for same-as clusters
            elif node.is_same_as_cluster():
                prototype = next(iter(node.get('prototype', shorten=False)), None)
                handle = next(iter(node.get('handle', shorten=True)), None)

                self.node_dict[str(node_label)] = SameAsClusterNode(
                    type='SameAsCluster',
                    prototype=str(prototype) if prototype else None,
                    handle=str(handle) if handle else None
                )

            # formatting for cluster membership
            elif node.is_cluster_membership():
                cluster = next(iter(node.get('cluster', shorten=False)), None)
                member = next(iter(node.get('clusterMember', shorten=False)), None)

                conf_levels = aida_graph.confidence_of(node_label)

                self.node_dict[str(node_label)] = ClusterMembershipNode(
                    type='ClusterMembership',
                    index=coref_counter,
                    cluster=str(cluster) if cluster else None,
                    clusterMember=str(member) if member else None,
                    conf=max(conf_levels) if conf_levels else None
                )

                coref_counter += 1

            # NEW COVID DOMAIN
            # formatting for claims
            elif node.is_claim():

                # strings
                claim_id = next(iter(node.get('claimId', shorten=False)), None)
                query_id = next(iter(node.get('queryId', shorten=False)), None)
                doc_id = next(iter(node.get('sourceDocument', shorten=False)), None)
                topic = next(iter(node.get('topic', shorten=False)), None)                
                subtopic = next(iter(node.get('subtopic', shorten=False)), None)                
                nl_description = next(iter(node.get('naturalLanguageDescription', shorten=False)), None)
                importance = next(iter(node.get('importance', shorten=False)), None)
                if importance is not None: importance = importance.toPython()
                epistemic = next(iter(node.get('epistemic', shorten=True)), None)
                sentiment = next(iter(node.get('sentiment', shorten=True)), None)

                logging.info("epistemic is {next(iter(node.get('epistemic', shorten=False)), None)}")
                logging.info("sentiment is {next(iter(node.get('sentiment', shorten=False)), None)}")
                

                # KB node IDs and node ID lists
                claim_semantics = list(iter(node.get('claimSemantics', shorten=False)))
                claim_template = list(iter(node.get('claimTemplate', shorten=False)))
                associated_kes = list(iter(node.get('associatedKEs', shorten=False)))
                xvar = list(iter(node.get('xVariable', shorten=False)))
                claimer = list(iter(node.get('claimer', shorten=False)))

                identical_claims = list(iter(node.get('identicalClaims', shorten=False)))
                related_claims = list(iter(node.get('relatedClaims', shorten=False)))
                supporting_claims = list(iter(node.get('supportingClaims', shorten=False)))
                refuting_claims = list(iter(node.get('refutingClaims', shorten=False)))

                self.node_dict[str(node_label)] = ClaimNode(
                    type = "Claim",
                    index = claim_counter,
                    claim_id = claim_id if claim_id else None,
                    doc_id = doc_id if doc_id else None,
                    query_id = query_id if query_id else None,
                    topic = topic if topic else None,
                    subtopic = subtopic if subtopic else None,
                    importance = importance if importance else None,
                    epistemic = str(epistemic) if epistemic else None,
                    sentiment = str(sentiment) if sentiment else None,
                    claim_template = claim_template if claim_template else None,
                    date_time = list(aida_graph.times_associated_with(node_label)),
                    text = nl_description if nl_description else None,
                    claim_semantics = claim_semantics,
                    associated_kes = associated_kes,
                    xvar = xvar,
                    claimer = claimer,
                    identical_claims = identical_claims,
                    related_claims = related_claims,
                    supporting_claims = supporting_claims,
                    refuting_claims = refuting_claims
                )

                claim_counter += 1
                self.claims.append(str(node_label))
                
            # NEW COVID DOMAIN
            # formatting for claim components
            elif node.is_claimcomponent():
                component_name = next(iter(node.get('componentName', shorten=False)), None)
                identity = next(iter(node.get('componentIdentity', shorten = False)), None)
                component_ke = next(iter(node.get('componentKE', shorten = False)), None)
                component_provenance = next(iter(node.get('componentProvenance', shorten = False)), None)
                component_type =  list(iter(node.get('componentType', shorten=False)))

                self.node_dict[str(node_label)] = ClaimComponentNode(
                    type='ClaimComponent',
                    index=claim_counter,
                    name=str(component_name) if component_name else None,
                    identity = identity if identity else None,
                    component_ke = component_ke if component_ke else None,
                    component_provenance = component_provenance if component_provenance else None,
                    component_type = component_type
                )

                claim_counter += 1
                
        logging.info('Done.')

    def validate_graph(self):
        logging.info('Validating the graph ...')

        # check if all clusters have handles. if they don't, add them.
        clusters_without_handles = set(
            node_label for node_label, node in self.node_dict.items()
            if node.type == 'SameAsCluster' and node.handle is None)

        if len(clusters_without_handles) > 0:
            # make a mapping from cluster IDs without handles to all names of all cluster members
            cluster_names_dict = defaultdict(list)

            # check the cluster nodes for the clusters without handles
            for cluster_label in clusters_without_handles:
                prototype_label = self.node_dict[cluster_label].prototype
                if prototype_label is not None and prototype_label in self.node_dict:
                    cluster_names_dict[cluster_label] += self.node_dict[prototype_label].name

            for node_label, node in self.node_dict.items():
                # check cluster membership nodes for the clusters without handles
                if node.type == 'ClusterMembership':
                    cluster_label, member_label = node.cluster, node.clusterMember
                    if cluster_label is not None and cluster_label in clusters_without_handles:
                        if member_label is not None and member_label in self.node_dict:
                            cluster_names_dict[cluster_label] += self.node_dict[member_label].name

            # now add a handle for all clusters that are missing one
            for cluster_label in clusters_without_handles:
                names = cluster_names_dict[cluster_label]
                if names:
                    self.node_dict[cluster_label].handle = min(names, key=lambda n: len(n))
                else:
                    self.node_dict[cluster_label].handle = '[unknown]'

    def as_dict(self) -> Dict:
        # noinspection PyProtectedMember
        return {
            'theGraph': {node_label: asdict(node)
                         for node_label, node in self.node_dict.items()},
            'ere': self.eres,
            'statements': self.statements
        }

    @classmethod
    def from_dict(cls, json_dict: Dict):
        # initialize the graph
        json_graph = cls()
        for node_label, node_dict in json_dict['theGraph'].items():
            node_type = node_dict.get('type')
            node_cls = node_cls_by_type.get(node_type, None)
            if not node_cls:
                raise RuntimeError(f'Unrecognized node type: {node_type}')

            # add the node to node_dict
            # noinspection PyArgumentList
            json_graph.node_dict[node_label] = node_cls(**node_dict)

        json_graph.eres = json_dict['ere']
        json_graph.statements = json_dict['statements']
        json_graph.claims = [key for key in json_dict["theGraph"] if json_dict["theGraph"][key]["type"] == "Claim"]

        json_graph.string_constants = list(json_graph.each_string_constant_of_graph())

        return json_graph

    # does this node exist in the graph
    def has_node(self, node_label):
        return node_label is not None and node_label in self.node_dict

    # is this node of a certain type (Entity, Event, Relation, Statement, etc)
    def is_node_type(self, node_label, node_type):
        return self.has_node(node_label) and self.node_dict[node_label].type == node_type

    def is_entity(self, node_label):
        return self.is_node_type(node_label, 'Entity')

    def is_event(self, node_label):
        return self.is_node_type(node_label, 'Event')

    def is_relation(self, node_label):
        return self.is_node_type(node_label, 'Relation')

    def is_ere(self, node_label):
        return self.is_entity(node_label) or self.is_event(node_label) or \
               self.is_relation(node_label)

    def is_claim(self, node_label):
        return self.is_node_type(node_label, "Claim")

    def is_statement(self, node_label):
        return self.is_node_type(node_label, 'Statement')

    # the subject of a statement node
    def stmt_subject(self, stmt_label):
        return self.node_dict[stmt_label].subject if self.is_statement(stmt_label) else None

    # the object of a statement node
    def stmt_object(self, stmt_label):
        return self.node_dict[stmt_label].object if self.is_statement(stmt_label) else None

    # the predicate of a statement node
    def stmt_predicate(self, stmt_label):
        return self.node_dict[stmt_label].predicate if self.is_statement(stmt_label) else None

    # subject or object of a statement node
    def stmt_arg_by_role(self, stmt_label, role_label):
        if self.is_statement(stmt_label):
            if role_label == 'subject':
                return self.stmt_subject(stmt_label)
            elif role_label == 'object':
                return self.stmt_object(stmt_label)
            else:
                logging.warning(f'Unknown role label: {role_label}')
        return None

    # arguments of a statement
    def stmt_args(self, stmt_label):
        if self.is_statement(stmt_label):
            return [self.node_dict[stmt_label].subject, self.node_dict[stmt_label].object]
        else:
            return []

    # if the node is a typing statement
    def is_type_stmt(self, stmt_label):
        return self.stmt_predicate(stmt_label) == 'type'

    # if the node is a statement on an event role
    def is_event_role_stmt(self, stmt_label):
        return self.is_event(self.stmt_subject(stmt_label)) and \
               self.is_ere(self.stmt_object(stmt_label))

    # if the node is a statement on a relation role
    def is_relation_role_stmt(self, stmt_label):
        return self.is_relation(self.stmt_subject(stmt_label)) and \
               self.is_ere(self.stmt_object(stmt_label))

    # iterate over EREs in the graph
    def each_ere(self):
        for node_label in self.eres:
            yield node_label, self.node_dict[node_label]

    # iterate over claims in the graph
    def each_claim(self):
        for node_label in self.claims:
            yield node_label, self.node_dict[node_label]

    # iterate over statements in the graph
    def each_statement(self):
        for node_label in self.statements:
            yield node_label, self.node_dict[node_label]

    # given an ERE label, iterate over labels of all adjacent statements, optionally, the statement
    # must have the given predicate, and the ERE must be the given role of the statement
    def each_ere_adjacent_stmt(self, ere_label, predicate=None, ere_role=None):
        if not self.is_ere(ere_label):
            return

        for stmt_label in self.node_dict[ere_label].adjacent:
            if not self.is_statement(stmt_label):
                continue
            if predicate is not None and self.stmt_predicate(stmt_label) != predicate:
                continue
            if ere_role is not None and self.stmt_arg_by_role(stmt_label, ere_role) != ere_label:
                continue
            yield stmt_label

    # possible types of an ERE
    def ere_types(self, ere_label):
        return set(self.shorten_label(self.node_dict[stmt_label].object)
                   for stmt_label in self.each_ere_adjacent_stmt(ere_label, 'type', 'subject'))

    # possible affiliations of an ERE: IDs of affiliation EREs
    def ere_affiliations(self, ere_label):
        return [self.stmt_object(stmt2) for stmt1, rel, stmt2
                in self.ere_affiliation_triples(ere_label)]

    # possible affiliation relations of an ERE: IDs of affiliations relations
    def ere_affiliation_relations(self, ere_label):
        return [rel for stmt1, rel, stmt2 in self.ere_affiliation_triples(ere_label)]

    # possible affiliation info of an ERE: each affiliation info is a triple of
    # (statement connecting the ERE to its affiliation relation,
    #  affiliation relation,
    #  statement connecting the affiliation relation to the affiliation)
    def ere_affiliation_triples(self, ere_label):
        affiliations = set()
        for stmt1 in self.each_ere_adjacent_stmt(ere_label, ere_role='object'):
            if self.is_affiliate_role_label(self.stmt_predicate(stmt1)):
                affiliation_rel = self.stmt_subject(stmt1)
                for stmt2 in self.each_ere_adjacent_stmt(affiliation_rel, ere_role='subject'):
                    if self.is_affiliation_role_label(self.stmt_predicate(stmt2)):
                        affiliations.add((stmt1, affiliation_rel, stmt2))
        return affiliations

    # is the ERE listed as an affiliation of some other EREs?
    def is_ere_affiliation(self, ere_label):
        for stmt in self.each_ere_adjacent_stmt(ere_label, ere_role='object'):
            if self.is_affiliation_role_label(self.stmt_predicate(stmt)):
                return True
        return False

    # if the label represent an affiliate in an affiliation relation
    @staticmethod
    def is_affiliate_role_label(label):
        if label.startswith('GeneralAffiliation') or label.startswith('OrganizationAffiliation'):
            if any(label.endswith(r) for r in affiliate_role_labels):
                return True
        return False

    # if the label represent an affiliation in an affiliation relation
    @staticmethod
    def is_affiliation_role_label(label):
        if label.startswith('GeneralAffiliation') or label.startswith('OrganizationAffiliation'):
            if any(label.endswith(r) for r in affiliation_role_labels):
                return True
        return False

    # possible names of an ERE, if any
    def ere_names(self, ere_label):
        if not self.is_ere(ere_label):
            return []
        return self.node_dict[ere_label].name

    # retain only names that are probably English
    @staticmethod
    def english_names(name_list):
        return [name for name in name_list if re.search(r'^[A-Za-z0-9\-,.\'\"()? ]+$', name)]

    # given a label, shorten it for easier reading
    @staticmethod
    def shorten_label(label):
        return label.split('/')[-1].split('#')[-1]

    # iterate over string constants in this graph (statement objects that are not nodes themselves)
    def each_string_constant_of_graph(self):
        for stmt_label, _ in self.each_statement():
            stmt_obj = self.stmt_object(stmt_label)
            if stmt_obj and stmt_obj not in self.node_dict:
                yield stmt_obj
                yield self.shorten_label(stmt_obj)

    def build_cluster_member_mappings(self, debug: bool = False):
        # Build mappings between clusters and members, and mappings between
        # clusters and prototypes
        print('\nBuilding mappings among clusters, members and prototypes ...')

        cluster_to_members = defaultdict(set)
        member_to_clusters = defaultdict(set)
        cluster_membership_key_mapping = defaultdict(set)

        cluster_to_prototype = {}
        prototype_to_clusters = defaultdict(set)

        for node_label, node in self.node_dict.items():
            if node.type == 'ClusterMembership':
                cluster, member = node.cluster, node.clusterMember
                assert cluster is not None and member is not None

                cluster_to_members[cluster].add(member)
                member_to_clusters[member].add(cluster)

                if debug and (cluster, member) in cluster_membership_key_mapping:
                    print('Warning: found duplicate ClusterMembership nodes for ({}, {})'.format(
                        cluster, member))
                cluster_membership_key_mapping[(cluster, member)].add(node_label)

            elif node.type == 'SameAsCluster':
                assert node_label not in cluster_to_prototype
                assert node.prototype is not None

                cluster_to_prototype[node_label] = node.prototype
                prototype_to_clusters[node.prototype].add(node_label)

                cluster_to_members[node_label].add(node.prototype)
                member_to_clusters[node.prototype].add(node_label)

        num_clusters = len(cluster_to_members)
        num_members = len(member_to_clusters)
        num_prototypes = len(prototype_to_clusters)
        assert len(cluster_to_prototype) == num_clusters

        print('\nConstructed mapping from {} clusters to {} members'.format(
            num_clusters, num_members))

        clusters_per_member_counter = Counter([len(v) for v in member_to_clusters.values()])
        for key in sorted(clusters_per_member_counter.keys()):
            if key > 1:
                print(
                    '\tFor {} out of {} members, each belong to {} clusters'.format(
                        clusters_per_member_counter[key], num_members, key))

        print('\nConstructed mapping from {} cluster-member pairs to '
              'their ClusterMembership node labels'.format(len(cluster_membership_key_mapping)))

        cm_node_per_pair_counter = Counter(len(v) for v in cluster_membership_key_mapping.values())
        for key in sorted(cm_node_per_pair_counter.keys()):
            if key > 1:
                print(
                    '\tFor {} out of {} cluster-member pairs, each is associated with '
                    '{} ClusterMembership nodes'.format(
                        cm_node_per_pair_counter[key], len(cluster_membership_key_mapping), key))

        print('\nConstructed mapping from {} clusters to {} prototypes'.format(
            num_clusters, num_prototypes))

        clusters_per_prototype_counter = Counter([len(v) for v in prototype_to_clusters.values()])
        for key in sorted(clusters_per_prototype_counter.keys()):
            if key > 1:
                print(
                    '\tFor {} out of {} prototypes, each is the prototype of {} '
                    'clusters'.format(
                        clusters_per_prototype_counter[key], num_prototypes, key))

        # Build mappings between members and prototypes, using the above
        # constructed mappings
        member_to_prototypes = defaultdict(set)
        prototype_to_members = defaultdict(set)

        for member, clusters in member_to_clusters.items():
            assert member not in member_to_prototypes
            for cluster in clusters:
                prototype = cluster_to_prototype[cluster]
                member_to_prototypes[member].add(prototype)
                prototype_to_members[prototype].add(member)

        assert len(member_to_prototypes) == num_members
        assert len(prototype_to_members) == num_prototypes

        print('\nConstructed mapping from {} members to {} prototypes'.format(
            num_members, num_prototypes))

        prototypes_per_member_counter = Counter([len(v) for v in member_to_prototypes.values()])
        for key in sorted(prototypes_per_member_counter.keys()):
            if key > 1:
                print(
                    '\tFor {} out of {} members, each is mapped to {} '
                    'prototypes'.format(
                        prototypes_per_member_counter[key], num_members, key))

        # Add ERE nodes that are not connected to any ClusterMembership node to
        # the mappings between members and prototypes. This shouldn't happen,
        # unless the TA2 output we get don't conform to the NIST-restricted
        # formatting requirements.
        ere_nodes_not_in_clusters = set()
        for node_label, node in self.node_dict.items():
            if node.type in ['Entity', 'Relation', 'Event']:
                if node_label not in member_to_prototypes:
                    ere_nodes_not_in_clusters.add(node_label)
        if len(ere_nodes_not_in_clusters) > 0:
            print('\nWarning: Found {} ERE nodes that are not connected to any '
                  'ClusterMembership node'.format(len(ere_nodes_not_in_clusters)))
            print('Adding them to the mappings between members and prototypes')
            for node_label in ere_nodes_not_in_clusters:
                member_to_prototypes[node_label].add(node_label)
                prototype_to_members[node_label].add(node_label)
            print(
                '\nAfter correction, constructed mapping from {} members to '
                '{} prototypes'.format(
                    len(member_to_prototypes), len(prototype_to_members)))

        mappings = {
            'cluster_to_members': cluster_to_members,
            'member_to_clusters': member_to_clusters,
            'cluster_membership_key_mapping': cluster_membership_key_mapping,
            'cluster_to_prototype': cluster_to_prototype,
            'prototype_to_clusters': prototype_to_clusters,
            'member_to_prototypes': member_to_prototypes,
            'prototype_to_members': prototype_to_members,
        }

        return mappings
