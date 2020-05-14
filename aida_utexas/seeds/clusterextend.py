# Katrin Erk March 2019
# Rule-based expansion of hypotheses


#########################
#########################
# class that manages cluster expansion
class ClusterExpansion:
    # initialize with an AidaJson object and an AidaHypothesisCollection 
    def __init__(self, graph_obj, hypothesis_obj):
        self.graph_obj = graph_obj
        self.hypothesis_obj = hypothesis_obj

    # compile json object that lists all the hypotheses with their statements
    def to_json(self):
        return self.hypothesis_obj.to_json()

    # make a list of strings with the clusters in readable form
    def to_s(self):
        return self.hypothesis_obj.to_s()

    # for each ERE, add *all* typing statements.
    # This is done using add_stmt and not extend (which calls filter)
    # because all type statements are currently thought to be compatible.
    # This code is not used anymore because the eval document wants to see only those event types
    # that match role labels
    def type_completion_useall(self):
        for hypothesis in self.hypothesis_obj.hypotheses:
            for ere_id in hypothesis.eres():
                for stmtlabel in self.graph_obj.each_ere_adjacent_stmt(ere_id, "type", "subject"):
                    hypothesis.add_stmt(stmtlabel)
            # add weights for the types we have been adding
            hypothesis.set_typeweights()

    # for each ERE, add typing statements.
    # For entities, add all. 
    # For events and relations, include only those types that match some role statement in the hypothesis.
    # This will still add multiple event types in case there are roles matching different event types.
    # It's up to the final filter to remove those if desired.
    def type_completion(self):
        # iterate over all hypotheses
        for hypothesis in self.hypothesis_obj.hypotheses:

            # print("HIER statements", hypothesis.stmts)
            # old_stmts = hypothesis.stmts.copy()
            
            # map each ERE to adjacent type statements
            ere_types = { }
            for ere_id in hypothesis.eres():
                if ere_id not in ere_types: ere_types[ ere_id ] = [ ]
                for stmtlabel in self.graph_obj.each_ere_adjacent_stmt(ere_id, "type", "subject"):
                    ere_types[ere_id].append(stmtlabel)

            # for each ERE in the hypothesis, add types matching a role statement
            for ere, typestmts in ere_types.items():
                if self.graph_obj.is_entity( ere):
                    # entity: add all types.
                    for stmt in typestmts:
                        hypothesis.add_stmt(stmt)
                        # print("entity type", stmt)
                else:
                    # event or relation: only add types that match a role included in the hypothesis
                    eventrel_roles_ere = [ self.graph_obj.shorten_label(rolelabel) for rolelabel, arg in hypothesis.eventrelation_each_argument(ere)]
                    for typestmt in typestmts:
                        typelabel = self.graph_obj.stmt_object(typestmt)
                        if typelabel is None:
                            # weird type statement, skip
                            continue

                        typelabel = self.graph_obj.shorten_label(typelabel)
                        if any(rolelabel.startswith(typelabel) and len(rolelabel) > len(typelabel) for rolelabel in eventrel_roles_ere):
                            # this is a type that is used in an outgoing edge of this ERE
                            hypothesis.add_stmt(typestmt)
                            # print("event type", typestmt)

            # print("HIER added statements", hypothesis.stmts.difference(old_stmts))
            # input("hit enter")

    # for each entity, add all affiliation statements.
    def affiliation_completion(self):
        # iterate over all hypotheses
        for hypothesis in self.hypothesis_obj.hypotheses:
            # iterate over EREs
            for ere in hypothesis.eres():
                # collect ERE IDs of all affiliation statements for this ERE
                possible_affiliations = set(self.graph_obj.stmt_object(stmt2) for stmt1, affrel, stmt2 in self.graph_obj.possible_affiliation_triples(ere))
                if len(possible_affiliations) == 1:
                    # possibly multiple affiliation statements, but all point to the same affiliation.
                    # add one of them. 
                    for stmt1, affiliationrel, stmt2 in self.graph_obj.possible_affiliation_triples(ere):
                        # print("HIER affiliation", ere, self.graph_obj.stmt_predicate(stmt1), stmt1, affiliationrel, stmt2, self.graph_obj.stmt_object(stmt2), self.graph_obj.thegraph[self.graph_obj.stmt_object(stmt2)].get("name", []))
                        hypothesis.add_stmt(stmt1)
                        hypothesis.add_stmt(stmt2)
                        break

    ## # unfinished!
    ## def relation_cmompletion(self):
    ##     # iterate over all hypotheses
    ##     for hypothesis in self.hypothesis_obj.hypotheses:
    ##         # iterate over EREs, find relations
    ##         for ere in hypothesis.eres():
    ##             if self.graph_obj.is_relation(ere):
    ##                 pass
            

    def hypotheses(self):
        return self.hypothesis_obj.hypotheses
