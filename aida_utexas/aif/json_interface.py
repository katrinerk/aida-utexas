################################
# Simple interface between an AidaGraph and Webppl:
# write out the graph as a big JavaScript data structure
# write out units for clustering:
# currently these are Statements that pertain to the same event mention


import json
import logging
from tqdm import tqdm
from rdflib.namespace import split_uri


###########
# Given an AidaGraph, transform it into input for WebPPL analysis:
# * re-encode the graph,
# * compute pairwise distances between statements
class JsonInterface:
    #def __init__(self, mygraph, entrypoints, simplification_level = 0, maxdist = 5):
    def __init__(self, mygraph, simplification_level = 0, maxdist = 5, compute_dist = False):
        self.mygraph = mygraph

        # main json object
        self.json_obj = { }
        # justification object
        self.json_just_obj = { }

        # re-encode the graph
        self.json_obj["theGraph"] = { }
        self.json_obj["ere"] = [ ]
        self.json_obj["statements"] = [ ]
        ## self.json_obj["coref_statements"] = [ ]
        self.statement_counter = 0
        self.ere_counter = 0
        self.coref_counter = 0

        # do the work
        self._transform_graph()

        # check if we have all the info we needed
        self._validate()
        
        # and pairwise statement distances. we consider maximal distances of 5.
        if compute_dist:
            self.dist = { }
            self.maxdist = maxdist


            # turn distance into proximity
            self.json_obj["statementProximity"] = self._compute_proximity()

        # complete the entry point information given 
        # self.json_obj["entrypoints"] = self._characterize_entrypoints(entrypoints)
        # self.json_obj["entrypoints"] = entrypoints

        # parameters for the one-class cluster generative model:
        # don't do that in this json object anymore
        ## self.json_obj["parameters"] = { "shape" :1.0, "scale" : 0.001 }
        ## self.json_obj["numSamples"] = 100
        ## self.json_obj["memberProb"] = 0.1

        # possibly simplify the model if we don't want to deal with the whole complexity of the data.
        # level 0 = no simplification
        # level 1 = fewer coref cluster statements
        # level 2 = no coref cluster statements
        self._simplify(simplification_level)

    # write main json output file
    def write(self, io):
        json.dump(self.json_obj, io, indent = 1)


    # write justifications
    def write_just(self, io):
        json.dump(self.json_just_obj, io, indent = 1)

    ###################################
    # functions that are actually doing the work

    def _transform_graph(self):
        logging.info('Transforming the graph...')

        self.json_obj["theGraph"] = { }
        self.json_obj["ere"] = [ ]
        self.json_obj["statements"] = [ ]
        ## self.json_obj["coref_statements"] = [ ]

        # this is where the justifications go:
        # node name -> justification
        self.json_just_obj = { }
        
        self.statement_counter = 0
        self.ere_counter = 0
        self.coref_counter = 0
        
        # we write out statements, events, entities, relations
        for node in tqdm(self.mygraph.nodes()):
            nodelabel = str(node.name)
            # entities, events, relations: they  have a type. They also have a list of adjacent statements,
            # and an index. They have optional names.
            if node.is_ere():
                self.json_obj["theGraph"][nodelabel] = {
                    "adjacent": self._adjacent_statements(node),
                    "index": self.ere_counter}

                if node.is_event():
                    self.json_obj["theGraph"][nodelabel]["type"] = "Event"
                elif node.is_entity():
                    self.json_obj["theGraph"][nodelabel]["type"] = "Entity"
                else:
                    self.json_obj["theGraph"][nodelabel]["type"] = "Relation"

                enames = list(set(self.mygraph.names_of_ere(node.name)))
                if len(enames) > 0:
                    self.json_obj["theGraph"][nodelabel]["name"] = enames

                # temporal information (for events)
                temporal = list(self.mygraph.times_associated_with(node.name))
                if len(temporal) > 0:
                    self.json_obj["theGraph"][nodelabel]["ldcTime"] = temporal

                self.ere_counter += 1
                self.json_obj["ere"].append(nodelabel)

                # record justification
                this_justification = self.get_justification(node)
                if len(this_justification) > 0:
                    self.json_just_obj[nodelabel] = this_justification
                
            # node describing a cluster: has a prototypical member and a handle (preferred name)
            elif node.is_sameas_cluster():
                self.json_obj["theGraph"][nodelabel] = {"type": "SameAsCluster"}
                
                content = node.get("prototype", shorten=False)
                if len(content) > 0:
                    self.json_obj["theGraph"][nodelabel]["prototype"] = str(content.pop())

                content = node.get("handle", shorten = True)
                if len(content) > 0:
                    self.json_obj["theGraph"][nodelabel]["handle"] = str(content.pop())
                    
                ## else:
                ##     # record this node only if it has a prototype as required
                ##     del self.json_obj["theGraph"][nodelabel]

            # clusterMembership statements have a cluster, a clusterMember, and a maximal confidence level
            elif node.is_cluster_membership():
                self.json_obj["theGraph"][nodelabel] = {
                    "type": "ClusterMembership",
                    "index": self.coref_counter}

                # cluster, clusterMember
                for label in ["cluster", "clusterMember"]:
                    content = node.get(label, shorten=False)
                    if len(content) > 0:
                        self.json_obj["theGraph"][nodelabel][label] = str(content.pop())
                

                # confidence
                conflevels = self.mygraph.confidence_of(node.name)
                if len(conflevels) > 0:
                    self.json_obj["theGraph"][ nodelabel]["conf"] = max(conflevels)

                self.coref_counter += 1
                
                ## # check that the node is well-formed
                ## if all(label in self.json_obj["theGraph"][nodelabel] for label in ["cluster", "clusterMember", "conf"]):
                ##     self.coref_counter += 1
                ##     # self.json_obj["coref_statements"].append(nodelabel)
                ## else:
                ##     del self.json_obj["theGraph"][nodelabel]
                  
                    
            # statements have a single subj, pred, obj, a maximal confidence level, and possibly mentions.
            # they also have hypotheses that they support, partially support, and contradict.
            # Statements also have justifications, which go in the justification object
            elif node.is_statement():
                # type
                self.json_obj["theGraph"][nodelabel] = {
                    "type": "Statement",
                    "index": self.statement_counter}

                # predicate, subject, object
                for label in ["subject", "object"]:
                    content = node.get(label, shorten=False)
                    if len(content) > 0:
                        self.json_obj["theGraph"][nodelabel][label] = str(content.pop())

                predicates = node.get("predicate", shorten=True)
                if len(predicates) > 0:
                    self.json_obj["theGraph"][nodelabel]["predicate"] = str(predicates.pop())

                # confidence
                conflevels = self.mygraph.confidence_of(node.name)
                if len(conflevels) > 0:
                    self.json_obj["theGraph"][ nodelabel]["conf"] = max(conflevels)

                ## # source document ids
                ## sources = set(self.mygraph.sources_associated_with(node.name))
                ## if len(sources) > 0:
                ##     self.json_obj["theGraph"][nodelabel]["source"] = list(sources)

                # hypotheses
                hypotheses = set(self.mygraph.hypotheses_supported(node.name))
                if len(hypotheses) > 0:
                    self.json_obj["theGraph"][nodelabel]["hypotheses_supported"] = list(hypotheses)
                hypotheses = set(self.mygraph.hypotheses_partially_supported(node.name))
                if len(hypotheses) > 0:
                    self.json_obj["theGraph"][nodelabel]["hypotheses_partially_supported"] = list(hypotheses)
                hypotheses = set(self.mygraph.hypotheses_contradicted(node.name))
                if len(hypotheses) > 0:
                    self.json_obj["theGraph"][nodelabel]["hypotheses_contradicted"] = list(hypotheses)

                self.statement_counter += 1
                ## # well-formedness check
                ## wellformed = False
                ## if all(label in self.json_obj["theGraph"][node.name] for label in ["conf", "predicate", "subject", "object"]):
                ##     wellformed = True
                ##     self.statement_counter += 1
                ##     self.json_obj["statements"].append(node.name)
                ## else:
                ##     del self.json_obj["theGraph"][node.name]

                # record justification
                ## if wellformed:
                ##     this_justification = self.get_justification(node)
                ##     if len(this_justification) > 0:
                ##         self.json_just_obj[node.name] = this_justification
                this_justification = self.get_justification(node)
                if len(this_justification) > 0:
                    self.json_just_obj[nodelabel] = this_justification
                    
                    

        ## # replace labels by label indices in adjacency statements
        ## for nodelabel in self.json_obj["theGraph"]:
        ##     if "adjacent" in self.json_obj["theGraph"][nodelabel]:
        ##         self.json_obj["theGraph"][nodelabel]["adjacent"] = [ self.json_obj["theGraph"][stmt]["index"] for stmt in self.json_obj["theGraph"][nodelabel]["adjacent"]]

        logging.info('Done.')


    def _validate(self):
        logging.info('Validating the graph...')

        # check if all clusters have handles. if they don't, add them.
        clusters_without_handles = set( label for label in self.json_obj["theGraph"]\
                                         if self.json_obj["theGraph"][label].get("type", None) == "SameAsCluster" and "handle" not in self.json_obj["theGraph"][label])

        if len(clusters_without_handles) > 0:
            # make a mapping from cluster IDs without handles to all names of all cluster members
            cluster_names = { }
            # check the cluster nodes for the clusters without handles
            for cluster in clusters_without_handles:
                member = self.json_obj["theGraph"][cluster].get("prototype", None)
                if member is not None and member in self.json_obj["theGraph"]:
                    if cluster not in cluster_names: cluster_names[ cluster ] = [ ]
                    cluster_names[ cluster] += self.json_obj["theGraph"][member].get("name", [])
                    
            for label in self.json_obj["theGraph"]:
                # check cluster membership nodes for the clusters without handles
                if self.json_obj["theGraph"][label].get("type", None) == "ClusterMembership" and self.json_obj["theGraph"][label].get("cluster", None) in clusters_without_handles:
                    cluster = self.json_obj["theGraph"][label].get("cluster", None)
                    member = self.json_obj["theGraph"][label].get("clusterMember", None)
                    if member is not None and member in self.json_obj["theGraph"]:
                        if cluster not in cluster_names: cluster_names[ cluster ] = [ ]
                        cluster_names[ cluster] += self.json_obj["theGraph"][member].get("name", [])

            # now add a handle for all clusters that are missing one
            for cluster in clusters_without_handles:
                if cluster in cluster_names and len(cluster_names[cluster]) > 0:
                    # grab the shortest name
                    if len(cluster_names[cluster]) > 0:
                        self.json_obj["theGraph"][cluster]["handle"] = min(cluster_names[cluster], key = lambda n:len(n))
                    else:
                        self.json_obj["theGraph"][cluster]["handle"] = "[unknown]"                        
                else:
                    self.json_obj["theGraph"][cluster]["handle"] = "[unknown]"
        

    # for an entity, relation, or event, determine all statements that mention it
    def _adjacent_statements(self, node):
        retv = set()

        # check all the neighbor nodes for whether they are statements
        for rel, neighbornodelabels in node.inedge.items():
            for neighbornodelabel in neighbornodelabels:

                neighbornode = self._getnode(neighbornodelabel)
                if neighbornode is not None:
                    if neighbornode.is_statement():
                        retv.add(neighbornode.name)
                        
        return list(retv)

    ## # for an entity, relation, or event, determine all cluster membership statements that it appears in.
    ## def _adjacent_sameas(self, node):
    ##     retv = set()

    ##     # check all the neighbor nodes for whether they are statements
    ##     for rel, neighbornodelabels in node.inedge.items():
    ##         for neighbornodelabel in neighbornodelabels:

    ##             neighbornode = self._getnode(neighbornodelabel)
    ##             if neighbornode is not None:
    ##                 if neighbornode.is_cluster_membership():
    ##                     # determine the name of the cluster
    ##                     retv.add(neighbornode.name)
                        
    ##     return list(retv)

    ## # for a SameAsCluster, determine all clusterMembership statements about it
    ## def _adjacent_members(self, node):
    ##     retv = set()

    ##     # check all the neighbor nodes for whether they are statements
    ##     for rel, neighbornodelabels in node.inedge.items():
    ##         for neighbornodelabel in neighbornodelabels:

    ##             neighbornode = self._getnode(neighbornodelabel)
    ##             if neighbornode is not None:
    ##                 if neighbornode.is_cluster_membership():
    ##                     retv.add(neighbornode.name)
                        
    ##     return list(retv)
    

    # compute the proximity between statements.
    # return as a dictionary mapping entry indices to dictionaries other_entry_index:proximity
    # proximities are normalized to sum to 1.
    def _compute_proximity(self):
        # get pairwise node distances
        # in the shape of a dictionary
        # self.dist: node label -> node label -> distance
        logging.info('Computing node distances...')
        self._compute_node_distances()
        logging.info('Done.')

        logging.info('Computing statement proximity...')
        # compute unit proximity
        retv = { }
        
        for stmt1 in tqdm(self.dist.keys()):
            # sum of proximities for reachable nodes
            sumprox = sum(self.maxdist - dist for dist in self.dist[stmt1].values())

            # proximity normalized by summed proximities
            if sumprox > 0:
                retv[stmt1] = { }
                for stmt2 in self.dist[stmt1].keys():
                    retv[stmt1][stmt2] = (self.maxdist - self.dist[stmt1][stmt2]) / sumprox

        logging.info('Done.')

        return retv
 

    # compute pairwise distances between graph nodes.
    # only start at statements, and go maximally self.maxdist nodes outward from each statement
    # don't use Floyd-Warshall, it's too slow with this many nodes
    def _compute_node_distances(self):
        # target data structure:
        # nodename1 -> nodename2 -> distance
        self.dist = { }

        # we only do statement nodes.
        self.statements = [ k for k, n in self.mygraph.node_dict.items() if n.is_statement()]
        # we step through neighbors that are EREs or statements
        visitlabels = set([ k for k, n in self.mygraph.node_dict.items() if n.is_statement() or n.is_ere() or\
                                n.is_sameas_cluster() or n.is_cluster_membership()])

        # do maxdist steps outwards from each statement node
        for subj in tqdm(self.statements):
            subjnode = self._getnode(subj)
            if subjnode is None: continue

            subjlabel = subjnode.name
            
            # next round of nodes to visit: neighbors of subj
            fringe = self._valid_neighbors(subjnode, visitlabels)

            dist = 1
            while dist < self.maxdist:
                newfringe = set()
                for obj in fringe:
                    if obj == subj: continue
                    objnode = self._getnode(obj)
                    if objnode is None: continue

                    objlabel = objnode.name
                    
                    if objnode.is_statement():
                        # keep track of distance only if this is a statement node
                        if subjlabel not in self.dist: self.dist[subjlabel] = { }
                        self.dist[subjlabel][objlabel] = min(self.dist[subjlabel].get(objlabel, self.maxdist + 1), dist)
                    
                    newfringe.update(self._valid_neighbors(objnode, visitlabels))
                fringe = newfringe
                dist += 1

                

    # # helper functions for node_distances. called by get_node_distances and _unit_distance
    # def getdist(self, l1, l2):
    #     if l1 == l2:
    #         return 0
    #     elif l1 < l2 and (l1, l2) in self.dist:
    #         return self.dist[ (l1, l2) ]
    #     elif l2 < l1 and (l2, l1) in self.dist:
    #         return self.dist[ (l2, l1) ]
    #     else: return self.unreachabledist

    # # sorted pair of label 1, label 2
    # def _nodepair(self, l1, l2):
    #     return tuple(sorted([l1, l2]))

    # return all node labels that have an incoming or outgoing edge to/from the node with label nodelabel
    def _valid_neighbors(self, node, visitlabels):
        retv = set()
        for nset in node.outedge.values(): retv.update(nset)
        for nset in node.inedge.values(): retv.update(nset)
        return retv.intersection(visitlabels)


    ## # prepare entry point descriptions to be in the right format for wppl.
    ## # an entry point is a dictionary with the following entries:
    ## # - ere: a list of labels of entities, relations, and events
    ## # - statements: a list of labels of statements
    ## # - corefacetLabels, corefacetFillers: two lists that together map labels of core facets to their fillers in ERE
    ## # (don't ask; it's because there is no way to create updated dictionaries in webppl where the key is stored in a variable)
    ## # - coreconstraints: a list of triples [ corefacetID, AIDAedgelabel, corefacetID] where
    ## #   statements corresponding to these triples need to be added to the cluster
    ## # - candidates: a list of statement labels such that these are exactly the statements that have one of the
    ## #   "ere" entries as one of their arguments. these are candidates for addition to the cluster.
    ## #
    ## # at this point we assume that the entry point has been completely filled in except for the "candidates".
    ## # This function modifies the entry points in place, adding candidates
    ## # and replacing each statement in "statements" by its ID
    ## def _characterize_entrypoints(self, entrypoints):
    ##     for entrypoint in entrypoints:

    ##         # fill in candidates            
    ##         candidateset = set()

    ##         # for each ERE in the entry point:
    ##         # its adjacent statements go in the set of candidates
    ##         for nodelabel in entrypoint["ere"]:
    ##             if nodelabel in self.json_obj["theGraph"] and \
    ##               self.json_obj["theGraph"][nodelabel]["type"] in ["Entity", "Event", "Relation"]:
    ##                 candidateset.update(self.json_obj["theGraph"][nodelabel]["adjacent"])

    ##         # statements that are already part of the entry point don't go into candidates
    ##         candidateset.difference_update(entrypoint["statements"])
            
    ##         entrypoint["candidates"] = list(candidateset)
            
    ##     return entrypoints

    def _getnode(self, nodelabel):
        if nodelabel in self.mygraph.node_dict:
            return self.mygraph.node_dict[nodelabel]
        else: return None
         

    def _potential_coref_duplicate(self, stmt1, stmt2, coref_dict):
        entry1 = self.json_obj["theGraph"][stmt1]
        entry2 = self.json_obj["theGraph"][stmt2]

        # have to have same predicate
        if entry1["predicate"] != entry2["predicate"]:
            return False

        for role in ["subject", "object"]:
            # not an entity or event? then the two entries need to be exactly the same
            if entry1[role] not in coref_dict:
                return entry1[role] == entry2[role]
            else:
                # entity or event? then they need to have a coref group in common
                return entry2[role] in coref_dict[ entry1[role ]]
            
    # determine cluster membership, and in each statement in theGraph,
    # add an entry that lists possible coref duplicates
    def _list_coref_duplicates(self):
        # map each cluster member to clusters, and each cluster to cluster members
        coref_member = { }
        member_coref = { }

        for label, entry in self.json_obj["theGraph"].items():
            if entry["type"] == "ClusterMembership":
                if entry["cluster"] not in coref_member: coref_member[entry["cluster"]] = set()
                coref_member[entry["cluster"]].add(entry["clusterMember"])
                if entry["clusterMember"] not in member_coref: member_coref[entry["clusterMember"]] = set()
                member_coref[entry["clusterMember"]].add(entry["cluster"])

        # map each cluster member to co-cluster members
        member_member = { }
        for ee in member_coref.keys():
            member_member[ ee ] = set()
            for coref in member_coref[ee]:
                member_member[ee].update(coref_member[coref])

        # now extend statement entries by duplicates
        for stmt in self.json_obj["statements"]:
            self.json_obj["theGraph"][stmt]["maybeCorefDuplicates"] = [ stmt2 for stmt2 in self.json_obj["statements"] if\
                                                                        stmt != stmt2 and\
                                                                        self._potential_coref_duplicate(stmt, stmt2, member_member) ]

            
    
    # simplify the graph so we have a simpler problem
    def _simplify(self, simplification_level, k = 2):
        if simplification_level == 2:
            # remove theGraph entries that are coref clusters or coref membership statements
            self.json_obj["theGraph"] = dict((key, value) for key, value in self.json_obj["theGraph"].items() if \
                                    self.json_obj["theGraph"][key]["type"] not in ["ClusterMembership", "SameAsCluster"])
            # delete list of coref statements
            # self.json_obj["coref_statements"] = [ ]

        elif simplification_level == 1:
            # for each coref cluster, only keep the coref membership of the prototype
            # and k other coref membership statements
            cluster_members = { }
            keep_coref_stmt = [ ]

            # determine which coref statements come from which clusters
            for key, value in self.json_obj["theGraph"].items():
                if value["type"] == "ClusterMembership":
                    if value["cluster"] not in cluster_members: cluster_members[value["cluster"]] = [ ]
                    cluster_members[value["cluster"]].append(key)
            
            # update coref cluster entries to reflect that
            for key in self.json_obj["theGraph"].keys():
                if self.json_obj["theGraph"][key]["type"] == "SameAsCluster":

                    value = self.json_obj["theGraph"][key]
                    # determine the statement label of the coref statement for the prototype
                    prototype_stmts = [cstmt for cstmt in cluster_members[key] if self.json_obj["theGraph"][cstmt]["clusterMember"] == value["prototype"]]
                    # determine the first k coref statements that are not the prototype statement
                    other_stmts = [cstmt for cstmt in cluster_members[key] if cstmt not in prototype_stmts][:k]

                    # remember that we are keeping these
                    keep_coref_stmt = keep_coref_stmt + prototype_stmts + other_stmts
                

            # update the graph
            # self.json_obj["coref_statements"] = keep_coref_stmt
            
            # update entity and event statements to only list coref statements that we are keeping
            for key in self.json_obj["theGraph"].keys():
                if self.json_obj["theGraph"][key]["type"] == "ClusterMember" and key not in keep_coref_stmt:
                    del self.json_obj["theGraph"][key]

    def get_justification(self, node):
        return list(self.mygraph.justifications_associated_with(node.name))

        ## self.json_just_obj[node.name] = [ ]
        ## textjustifications = list(self.mygraph.sources_and_textjust_associated_with(node.name))
        ## if len(textjustifications) > 0:
        ##     self.json_just_obj[node.name] = textjustifications                

    def simplify_subsubtypes(self, type_mapping=None, role_mapping=None):
        for stmt_name, stmt_node in self.json_obj['theGraph'].items():
            if stmt_node['type'] == 'Statement':
                pred_name = stmt_node.get('predicate', None)

                if pred_name == 'type':
                    type_str = stmt_node.get('object', None)
                    if type_str is not None:
                        type_namespace, type_name = split_uri(type_str)

                        new_type_name = None

                        if type_mapping is not None and type_name in type_mapping:
                            new_type_name = type_mapping[type_name]

                        elif len(type_name.split('.')) > 2:
                            assert len(type_name.split('.')) == 3
                            new_type_name, subsubtype = type_name.rsplit('.', maxsplit=1)
                            assert '_' not in subsubtype

                        if new_type_name is not None and new_type_name != type_name:
                            new_type_str = type_namespace + new_type_name
                            stmt_node['object'] = new_type_str
                            stmt_node['object_original'] = type_str

                elif pred_name is not None:
                    new_pred_name = None

                    if role_mapping is not None and pred_name in role_mapping:
                        new_pred_name = role_mapping[pred_name]

                    elif len(pred_name.split('.')) > 2:
                        assert len(pred_name.split('.')) == 3

                        new_pred_name, subsubtype_w_role = pred_name.rsplit('.', maxsplit=1)
                        assert len(subsubtype_w_role.split('_')) == 2
                        role_name = subsubtype_w_role.split('_')[1]
                        new_pred_name = '{}_{}'.format(new_pred_name, role_name)

                    if new_pred_name is not None and new_pred_name != pred_name:
                        stmt_node['predicate'] = new_pred_name
                        stmt_node['predicate_original'] = pred_name
