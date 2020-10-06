import math
from argparse import ArgumentParser
from io import BytesIO
from operator import itemgetter

from rdflib import Graph
from rdflib.namespace import Namespace, RDF, XSD, NamespaceManager
from rdflib.plugins.serializers.turtle import TurtleSerializer
from rdflib.plugins.serializers.turtle import VERB
from rdflib.term import BNode, Literal, URIRef

from aida_utexas import util
from aida_utexas.aif import JsonGraph
from pipeline.rdflib_helper import AIDA, LDC, LDC_ONT
from pipeline.rdflib_helper import catalogue_kb_nodes
from pipeline.rdflib_helper import index_statement_nodes, index_cluster_membership_nodes
from pipeline.rdflib_helper import index_type_statement_nodes
from pipeline.rdflib_helper import triples_for_cluster, triples_for_cluster_membership
from pipeline.rdflib_helper import triples_for_edge_stmt, triples_for_type_stmt, triples_for_ere

UTEXAS = Namespace('http://www.utexas.edu/aida/')

expanding_preds_for_stmt = \
    [AIDA.justifiedBy, AIDA.confidence, AIDA.containedJustification, AIDA.boundingBox, AIDA.system]
excluding_preds_for_stmt = [AIDA.privateData]

expanding_preds_for_ere = \
    [AIDA.informativeJustification, AIDA.confidence, AIDA.boundingBox, AIDA.system]
excluding_preds_for_ere = [AIDA.privateData, AIDA.link, AIDA.justifiedBy, AIDA.ldcTime]

expanding_preds_for_cluster = \
    [AIDA.informativeJustification, AIDA.confidence, AIDA.boundingBox, AIDA.system]
excluding_preds_for_cluster = [AIDA.privateData, AIDA.link, AIDA.justifiedBy, AIDA.ldcTime]

expanding_preds_for_cm = [AIDA.confidence, AIDA.system]
excluding_preds_for_cm = []


# trying to match the AIF format
class AIFSerializer(TurtleSerializer):
    xsd_namespace_manager = NamespaceManager(Graph())
    xsd_namespace_manager.bind('xsd', XSD)

    # when writing BNode as subjects, write closing bracket
    # in a new line at the end
    def s_squared(self, subject):
        if (self._references[subject] > 0) or not isinstance(subject, BNode):
            return False
        self.write('\n' + self.indent() + '[')
        self.predicateList(subject)
        self.write('\n] .')
        return True

    # when printing Literals, directly call Literal.n3()
    def label(self, node, position):
        if node == RDF.nil:
            return '()'
        if position is VERB and node in self.keywords:
            return self.keywords[node]
        if isinstance(node, Literal):
            return node.n3(namespace_manager=self.xsd_namespace_manager)
        else:
            node = self.relativize(node)

            return self.getQName(node, position == VERB) or node.n3()


# return the text representation of the graph
def print_graph(g):
    serializer = AIFSerializer(g)
    stream = BytesIO()
    serializer.serialize(stream=stream)
    return stream.getvalue().decode()


def compute_handle_mapping(ere_set, json_graph, member_to_clusters, cluster_to_prototype):
    entity_cluster_set = set()
    for ere in ere_set:
        # Only include handles for clusters of Entities.
        if json_graph.is_entity(ere):
            for cluster in member_to_clusters[ere]:
                entity_cluster_set.add(cluster)

    proto_handles = {}
    for cluster in entity_cluster_set:
        prototype = cluster_to_prototype.get(cluster, None)
        if prototype is not None:
            proto_handles[prototype] = json_graph.node_dict[cluster].handle

    return proto_handles


# build a subgraph from a list of statements for one AIDA result
def build_subgraph_for_hypothesis(kb_graph, kb_nodes_by_category, kb_stmt_key_mapping,
                                  kb_cm_key_mapping, kb_type_stmt_key_mapping, json_graph,
                                  graph_mappings, hypothesis, hypothesis_id, hypothesis_weight):
    member_to_clusters = graph_mappings['member_to_clusters']
    cluster_to_prototype = graph_mappings['cluster_to_prototype']

    # Set of all KB edge statement nodes
    kb_edge_stmt_set = set()
    # Mapping from ERE to all its KB type statement nodes
    kb_type_stmt_set = set()

    # Mapping from KB edge statement nodes to importance values
    kb_stmt_importance = {}

    # Set of all ERE node labels
    ere_set = set()
    # Mapping from ERE node labels to importance values
    ere_importance = {}

    # logging.info('Processing all statements')
    for stmt_label, stmt_weight in zip(hypothesis['statements'], hypothesis['statementWeights']):
        # Rescale the stmt_weight to get the importance value
        if stmt_weight <= 0.0:
            stmt_weight = math.exp(stmt_weight / 100.0)
        else:
            stmt_weight = 0.0001

        assert json_graph.is_statement(stmt_label)
        stmt_entry = json_graph.node_dict[stmt_label]

        stmt_subj = stmt_entry.subject
        stmt_pred = stmt_entry.predicate
        stmt_obj = stmt_entry.object
        assert stmt_subj is not None and stmt_pred is not None and stmt_obj is not None

        # Find the statement node in the KB
        kb_stmt_id = URIRef(stmt_label)
        if kb_stmt_id not in kb_nodes_by_category['Statement']:
            kb_stmt_pred = RDF.type if stmt_pred == 'type' else LDC_ONT.term(stmt_pred)
            kb_stmt_id = next(iter(
                kb_stmt_key_mapping[(URIRef(stmt_subj), kb_stmt_pred, URIRef(stmt_obj))]))

        # Add the subject of any statement to ere_set
        ere_set.add(stmt_subj)

        # Update the importance value of the subject of any statement based on stmt_weight
        if stmt_subj not in ere_importance or ere_importance[stmt_subj] < stmt_weight:
            ere_importance[stmt_subj] = stmt_weight

        if stmt_pred == 'type':
            if kb_stmt_id is not None:
                # Add kb_stmt_id to the set of KB type statement nodes
                kb_type_stmt_set.add(kb_stmt_id)
                # kb_type_stmt_dict[stmt_subj].add(kb_stmt_id)

        else:
            if kb_stmt_id is not None:
                # Add kb_stmt_id to the set of KB edge statement nodes
                kb_edge_stmt_set.add(kb_stmt_id)
                # Update the importance value of the edge statement
                kb_stmt_importance[kb_stmt_id] = stmt_weight

            # Add the object of edge statements to ere_set
            ere_set.add(stmt_obj)

            # Update the importance value of the object of edge statements based on stmt_weight
            if stmt_obj not in ere_importance or ere_importance[stmt_obj] < stmt_weight:
                ere_importance[stmt_obj] = stmt_weight

    # Set of all SameAsCluster node labels
    same_as_cluster_set = set()
    # Set of all KB ClusterMembership nodes
    kb_cluster_membership_set = set()

    # Set of all ERE node labels that are prototypes
    proto_ere_set = set()
    # Mapping from ERE prototype node labels to importance values
    proto_importance = {}

    # logging.info('Processing all EREs and clusters')
    cluster_memberships = hypothesis.get('clusterMemberships', None)
    if cluster_memberships is None:
        for ere in ere_set:
            ere_weight = ere_importance.get(ere, 0.0)
            for cluster in member_to_clusters[ere]:
                # Add all corresponding cluster label of each ERE node to same_as_cluster_set
                same_as_cluster_set.add(cluster)

                # Find the ClusterMembership node in the KB
                kb_cluster_membership_set.update(kb_cm_key_mapping[URIRef(cluster), URIRef(ere)])

                proto_ere = cluster_to_prototype[cluster]
                if proto_ere not in proto_importance or proto_importance[proto_ere] < ere_weight:
                    proto_importance[proto_ere] = ere_weight
    else:
        for member, cluster in cluster_memberships:
            same_as_cluster_set.add(cluster)
            kb_cluster_membership_set.update(kb_cm_key_mapping[URIRef(cluster), URIRef(member)])

            # Add the prototype of each SameAsCluster node to ere_set
            proto_ere = cluster_to_prototype[cluster]
            proto_ere_set.add(proto_ere)

            # Find the type statement node for the prototype
            proto_type_stmt_id_list = kb_type_stmt_key_mapping[URIRef(proto_ere)]
            highest_granularity_level = max(
                [len(type_ont.split('.')) for _, type_ont in proto_type_stmt_id_list])
            for type_stmt_id, type_ont in proto_type_stmt_id_list:
                if len(type_ont.split('.')) == highest_granularity_level:
                    kb_type_stmt_set.add(type_stmt_id)

            # Find the ClusterMembership node for the prototype in the KB
            kb_cluster_membership_set.update(kb_cm_key_mapping[URIRef(cluster), URIRef(proto_ere)])

            member_weight = ere_importance.get(member, 0.0)
            if proto_ere not in proto_importance or proto_importance[proto_ere] < member_weight:
                proto_importance[proto_ere] = member_weight

    # Add all prototype ERE labels to ere_set
    ere_set |= proto_ere_set

    # All triples to be added to the subgraph
    # logging.info('Extracting all content triples')
    all_triples = set()

    for kb_stmt_id in kb_edge_stmt_set:
        all_triples.update(triples_for_edge_stmt(kb_graph, kb_stmt_id))

    for kb_stmt_id in kb_type_stmt_set:
        all_triples.update(triples_for_type_stmt(kb_graph, kb_stmt_id))

    # logging.info('Extracting triples for all EREs')
    # Add triples for all EREs
    for ere in ere_set:
        kb_ere_id = URIRef(ere)
        all_triples.update(triples_for_ere(kb_graph, kb_ere_id))

    # logging.info('Extracting triples for all SameAsClusters')
    # Add triples for all SameAsClusters
    for cluster in same_as_cluster_set:
        kb_cluster_id = URIRef(cluster)
        all_triples.update(triples_for_cluster(kb_graph, kb_cluster_id))

    # logging.info('Extracting triples for all ClusterMemberships')
    # Add triples for all ClusterMemberships
    for kb_cm_id in kb_cluster_membership_set:
        all_triples.update(triples_for_cluster_membership(kb_graph, kb_cm_id))

    # logging.info('Constructing a subgraph')
    # Start building the subgraph
    subgraph = Graph()

    # Bind all prefixes of kb_graph to the subgraph
    for prefix, namespace in kb_graph.namespaces():
        if str(namespace) not in [AIDA, LDC, LDC_ONT]:
            subgraph.bind(prefix, namespace)
    # Bind the AIDA, LDC, LDC_ONT, and UTEXAS namespaces to the subgraph
    subgraph.bind('aida', AIDA, override=True)
    subgraph.bind('ldc', LDC, override=True)
    subgraph.bind('ldcOnt', LDC_ONT, override=True)
    subgraph.bind('utexas', UTEXAS)

    # logging.info('Adding hypothesis related triples to the subgraph')
    # Add triple for the aida:Hypothesis node and its type
    kb_hypothesis_id = UTEXAS.term(hypothesis_id)
    subgraph.add((kb_hypothesis_id, RDF.type, AIDA.Hypothesis))

    # Add triple for the hypothesis importance value
    subgraph.add((kb_hypothesis_id, AIDA.importance,
                  Literal(hypothesis_weight, datatype=XSD.double)))

    # Add triple for the aida:Subgraph node and its type
    kb_subgraph_id = UTEXAS.term(hypothesis_id + '_subgraph')
    subgraph.add((kb_hypothesis_id, AIDA.hypothesisContent, kb_subgraph_id))
    subgraph.add((kb_subgraph_id, RDF.type, AIDA.Subgraph))

    # Add all EREs as contents of the aida:Subgraph node
    for ere in ere_set:
        kb_ere_id = URIRef(ere)
        subgraph.add((kb_subgraph_id, AIDA.subgraphContains, kb_ere_id))

    # logging.info('Adding all content triples to the subgraph')
    # Add all triples
    for triple in all_triples:
        subgraph.add(triple)

    # Add importance values for all edge statements
    for kb_stmt_id, importance in kb_stmt_importance.items():
        subgraph.add((kb_stmt_id, AIDA.importance, Literal(importance, datatype=XSD.double)))

    # Add importance values for all prototype EREs
    for proto_ere, proto_weight in proto_importance.items():
        kb_proto_id = URIRef(proto_ere)
        subgraph.add((kb_proto_id, AIDA.importance, Literal(proto_weight, datatype=XSD.double)))

    # Compute handles for Entity clusters
    proto_handles = compute_handle_mapping(
        ere_set, json_graph, member_to_clusters, cluster_to_prototype)

    for proto_ere, handle in proto_handles.items():
        kb_proto_id = URIRef(proto_ere)
        if len(list(subgraph.objects(subject=kb_proto_id, predicate=AIDA.handle))) == 0:
            subgraph.add((kb_proto_id, AIDA.handle, Literal(handle, datatype=XSD.string)))

    return subgraph


def main():
    parser = ArgumentParser()
    parser.add_argument('graph_path', help='path to the graph json file')
    parser.add_argument('hypotheses_path', help='path to the hypotheses json directory')
    parser.add_argument('kb_path', help='path to the TA2 KB file (in AIF)')
    parser.add_argument('output_dir', help='path to output directory')
    parser.add_argument('run_id', help='TA3 run ID')
    parser.add_argument('sin_id_prefix', help='prefix of SIN IDs to name the final hypotheses')
    parser.add_argument('--top', default=50, type=int,
                        help='number of top hypothesis to output')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='If specified, overwrite existing output files without warning')

    args = parser.parse_args()

    json_graph = JsonGraph.from_dict(util.read_json_file(args.graph_path, 'JSON graph'))

    graph_mappings = json_graph.build_cluster_member_mappings()

    hypotheses_file_paths = util.get_file_list(args.hypotheses_path, suffix='.json', sort=True)

    print('Reading kb from {}'.format(args.kb_path))
    kb_graph = Graph()
    kb_graph.parse(args.kb_path, format='ttl')

    kb_nodes_by_category = catalogue_kb_nodes(kb_graph)

    kb_stmt_key_mapping = index_statement_nodes(
        kb_graph, kb_nodes_by_category['Statement'])
    kb_cm_key_mapping = index_cluster_membership_nodes(
        kb_graph, kb_nodes_by_category['ClusterMembership'])
    kb_type_stmt_key_mapping = index_type_statement_nodes(
        kb_graph, kb_nodes_by_category['TypeStatement'])

    output_dir = util.get_output_dir(args.output_dir, overwrite_warning=not args.force)

    run_id = args.run_id
    sin_id_prefix = args.sin_id_prefix

    for hypotheses_file_path in hypotheses_file_paths:
        hypotheses_json = util.read_json_file(hypotheses_file_path, 'hypotheses')

        print('Found {} hypotheses with probability {}'.format(
            len(hypotheses_json['probs']), hypotheses_json['probs']))

        soin_id = sin_id_prefix + hypotheses_file_path.stem.split('_')[0]
        frame_id = soin_id + '_F1'

        top_count = 0
        for hypothesis_idx, prob in sorted(
                enumerate(hypotheses_json['probs']), key=itemgetter(1), reverse=True):
            if prob <= 0.0:
                hypothesis_weight = math.exp(prob / 2.0)
            else:
                hypothesis_weight = 0.0001

            hypothesis = hypotheses_json['support'][hypothesis_idx]

            top_count += 1
            hypothesis_id = '{}_hypothesis_{:0>3d}'.format(frame_id, top_count)

            subgraph = build_subgraph_for_hypothesis(
                kb_graph=kb_graph,
                kb_nodes_by_category=kb_nodes_by_category,
                kb_stmt_key_mapping=kb_stmt_key_mapping,
                kb_cm_key_mapping=kb_cm_key_mapping,
                kb_type_stmt_key_mapping=kb_type_stmt_key_mapping,
                json_graph=json_graph,
                graph_mappings=graph_mappings,
                hypothesis=hypothesis,
                hypothesis_id=hypothesis_id,
                hypothesis_weight=hypothesis_weight
            )

            output_path = output_dir / '{}.{}.{}.H{:0>3d}.ttl'.format(
                run_id, soin_id, frame_id, top_count)
            print('Writing hypothesis #{} with prob {} to {}'.format(top_count, prob, output_path))
            with open(output_path, 'w') as fout:
                fout.write(print_graph(subgraph))

            if top_count >= args.top:
                break


if __name__ == '__main__':
    main()
