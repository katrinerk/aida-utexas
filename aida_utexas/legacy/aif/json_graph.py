################################
# Tools for working with json format AIDA graphs


import re
import graphviz


class AidaJson:
    def __init__(self, json_obj):
        self.json_obj = json_obj
        self.thegraph = json_obj["theGraph"]
        # make a set of all string constants that appear as statement arguments
        self.string_constants_of_graph = set(self._each_string_constant_of_graph())

    ###############################

    ###
    # iterate over Entities, Events, Relations, EREs, Statements in the graph
    def each_ere(self):
        for nodelabel, node in self.thegraph.items():
            if node["type"] in ["Entity", "Event", "Relation"]:
                yield (nodelabel, node)

    def each_entity(self):
        for nodelabel, node in self.thegraph.items():
            if node["type"] == "Entity":
                yield (nodelabel, node)

    def each_event(self):
        for nodelabel, node in self.thegraph.items():
            if node["type"] == "Event":
                yield (nodelabel, node)

    def each_relation(self):
        for nodelabel, node in self.thegraph.items():
            if node["type"] == "Relation":
                yield (nodelabel, node)

    def each_statement(self):
        for nodelabel, node in self.thegraph.items():
            if node["type"] == "Statement":
                yield (nodelabel, node)

    # arguments of a statement
    def statement_args(self, stmtlabel):
        if stmtlabel not in self.thegraph or not self.is_statement(stmtlabel):
            return [ ]
        else:
            return [ self.thegraph[stmtlabel][role] for role in ["subject", "object"]]

    # is this node an Entity/Event/Relation/Statement
    def is_nodetype(self, nodelabel, nodetype):
        return (nodelabel in self.thegraph and self.thegraph[nodelabel]["type"] == nodetype)

    def is_node(self, nodelabel):
        return nodelabel in self.thegraph
    
    def is_statement(self, nodelabel):
        return self.is_nodetype(nodelabel, "Statement")

    def is_entity(self, nodelabel):
        return self.is_nodetype(nodelabel, "Entity")

    def is_event(self, nodelabel):
        return self.is_nodetype(nodelabel, "Event")

    def is_relation(self, nodelabel):
        return self.is_nodetype(nodelabel, "Relation")
                
    def is_ere(self, nodelabel):
        return self.is_entity(nodelabel) or self.is_event(nodelabel) or self.is_relation(nodelabel)

    def is_typestmt(self, stmtlabel):
        return self.is_statement(stmtlabel) and self.thegraph[stmtlabel]["predicate"] == "type"

    def is_eventrole_stmt(self, stmtlabel):
        return self.is_statement(stmtlabel) and self.is_event(self.stmt_subject(stmtlabel)) and self.is_ere(self.stmt_object(stmtlabel))

    def is_relationrole_stmt(self, stmtlabel):
        return self.is_statement(stmtlabel) and self.is_relation(self.stmt_subject(stmtlabel)) and self.is_ere(self.stmt_object(stmtlabel))
    
    def stmt_subject(self, stmtlabel):
        if not self.is_statement(stmtlabel):
            return None
        else: return self.thegraph[stmtlabel]["subject"]

    def stmt_object(self, stmtlabel):
        if not self.is_statement(stmtlabel):
            return None
        else: return self.thegraph[stmtlabel]["object"]

    def stmt_predicate(self, stmtlabel):
        if not self.is_statement(stmtlabel):
            return None
        else: return self.thegraph[stmtlabel]["predicate"]

    ###
    # given an ERE label, return labels of all adjacent statements
    # with the given predicate and where erelabel is an argument in the given ererole (subject, object)
    def each_ere_adjacent_stmt(self, erelabel, predicate, ererole):
        for stmtlabel in self.each_ere_adjacent_stmt_anyrel(erelabel):
            if self.thegraph[stmtlabel][ererole] == erelabel and \
              self.thegraph[stmtlabel]["predicate"] == predicate:
                yield stmtlabel

    ###
    # given an ERE label, return labels of all adjacent statements
    def each_ere_adjacent_stmt_anyrel(self, erelabel):
        if erelabel not in self.thegraph:
            return

        for stmtlabel in self.thegraph[erelabel].get("adjacent", []):
            if stmtlabel in self.thegraph:
                yield stmtlabel
        

    ###
    # possible types of an ERE: strings
    def possible_types(self, erelabel):
        return set(self.shorten_label(self.thegraph[stmtlabel]["object"]) \
                       for stmtlabel in self.each_ere_adjacent_stmt(erelabel, "type", "subject"))

    ###
    # possible affiliations of an ERE: IDs of affiliation EREs
    def possible_affiliations(self, erelabel):
        return [ self.stmt_subject(stmt2) for stmt1, rel, stmt2 in self.possible_affiliation_triples(erelabel) ]

    ###
    # possible affiliation relations of an ERE: IDs of affiliations relations in which the ERE is the affiliate
    def possible_affiliation_relations(self, erelabel):
        return [rel for stmt1, rel, stmt2 in self.possible_affiliation_triples(erelabel)]

    ###
    # affiliation info for an ERE:
    # triples of
    # (statement connecting the ERE to its affiliation relation,
    #  affiliation relation,
    # statement connecting the affiliation relation to the affiliation)
    def possible_affiliation_triples(self, erelabel):
        affiliations = set()
        for stmt1 in self.each_ere_adjacent_stmt_anyrel(erelabel):
            if self.stmt_object(stmt1) == erelabel and self.is_affiliate_rolelabel(self.stmt_predicate(stmt1)):
                affiliation_rel = self.stmt_subject(stmt1)
                for stmt2 in self.each_ere_adjacent_stmt_anyrel(affiliation_rel):
                    if self.stmt_subject(stmt2) == affiliation_rel and self.is_affiliation_rolelabel(self.stmt_predicate(stmt2)):
                        affiliations.add( (stmt1, affiliation_rel, stmt2))

        return affiliations
        
    
    ####
    # names, if any
    def ere_names(self, erelabel):
        names = self.thegraph[erelabel].get("name", [])
        if names != []:
            return names
        
        names = self.thegraph[erelabel].get("hasName", [ ])
        return names
            
    
    ######################################

    ###
    # return a dictionary that characterizes the given ERE in terms of:
    # - ere type (nodetype)
    # - names ("name")
    # - type statements associated ("typestmt")
    # - affiliation
    def ere_characterization(self, erelabel):
        retv = { }

        if erelabel in self.thegraph:
            retv["label"] = erelabel
            retv["nodetype"] = self.shorten_label(self.thegraph[erelabel]["type"])

            retv["typestmt"] = ", ".join(self.possible_types(erelabel))

            retv["name"] = ", ".join(self.english_names(self.thegraph[erelabel].get("name", [])))

            affiliations = set()
            for affiliationlabel in self.possible_affiliations(erelabel):
                if "name" in self.thegraph[affiliationlabel]:
                    affiliations.update(self.thegraph[affiliationlabel]["name"])        


            retv["affiliation"] = ", ".join(self.english_names(affiliations))

        return retv

    ####
    def print_ere_characterization(self, erelabel, fout, short=False):
        characterization = self.ere_characterization(erelabel)
        if short:
            print("\t label :", characterization["label"], file = fout)
        else:
            for key in ["label", "nodetype", "name", "typestmt", "affiliation"]:
                if key in characterization and characterization[key] != "":
                    print("\t", key, ":", characterization[key], file = fout)
                    
    ####
    # print characterization of a given statement in terms of:
    # predicate, subject, object
    # subject and object can be strings or ERE characterizations
    def print_statement_info(self, stmtlabel, fout, additional = None):
        if stmtlabel not in self.thegraph:
            return

        node = self.thegraph[stmtlabel]

        print("---", file = fout)
        print("Statement", stmtlabel, file = fout)
        for label in ["subject", "predicate", "object"]:
            if node[label] in self.thegraph:
                print(label, ":", file = fout)
                self.print_ere_characterization(node[label], fout, short = (node["predicate"] == "type"))
            else:
                print(label, ":", self.shorten_label(node[label]), file = fout)
        if additional:
            print("---", additional, "---", file = fout)
        print("\n", file = fout)

    ####
    # Given a set of statement labels, sort the labels for more human-friendly output:
    # group all statements that refer to the same event
    def sorted_statements_for_output(self, stmtset):
        # map from event labels to statement that mention them
        event_stmt = { }
        for stmtlabel in stmtset:
            node = self.thegraph.get(stmtlabel, None)
            if node is None: continue
            for rel in ["subject", "object"]:
                if node[rel] in self.thegraph and self.thegraph[node[rel]].get("type", None) == "Event":
                    if node[rel] not in event_stmt:
                        event_stmt[ node[rel]] = set()
                    event_stmt[ node[rel] ].add(stmtlabel)

        # put statements in output list in order of events that mention them
        stmts_done = set()
        retv = [ ]
        for stmts in event_stmt.values():
            for stmt in stmts:
                if stmt not in stmts_done:
                    stmts_done.add(stmt)
                    retv.append(stmt)

        # and statements that don't mention events
        for stmt in stmtset:
            if stmt not in stmts_done:
                stmts_done.add(stmt)
                retv.append(stmt)

        return retv


    ################################
    # do visualization with graphviz
    def graphviz(self, stmt = None, outfilename = None, showview = False, unary_stmt = False):

        ##
        # restrict the statements to include in the visualization?
        # if so, make a list of EREs to include in the visualization
        if stmt is not None:
            ere_to_include = set()
            for nodelabel, node in self.each_statement():
                if nodelabel in stmt:
                    for argument in ["subject", "object"]:
                        if self.is_ere(node[argument]):
                            ere_to_include.add(node[argument])

        ##
        # now visualize
        dot = graphviz.Digraph(comment = "AIDA graph", engine = "circo")

        # color scheme
        colorscheme = {
            "Entity" : "beige",
            "Event" : "lightblue",
            "Relation": "lightgrey"
        }

        # make all the ERE nodes
        for nodelabel, node in self.each_ere():
            # are we only visualizing part of the graph?
            if stmt is not None and nodelabel not in ere_to_include:
                # skip this ere
                continue

            characterization = self.ere_characterization(nodelabel)
            erecolor = colorscheme[ node["type"]]

            nodecontent = ""
            if "typestmt" in characterization and characterization["typestmt"] != "":
                nodecontent += "type:" + characterization["typestmt"] + "\n"

            if "name" in characterization and characterization["name"] != "":
                nodecontent += characterization["name"]
        
            dot.node(self.shorten_label(nodelabel),  nodecontent, color=erecolor, style = "filled")


        # make statements into connections
        for nodelabel, node in self.each_statement():
            # are we only visualizing part of the graph
            if stmt is not None and nodelabel not in stmt:
                continue
            
            # statements that connects two EREs
            if self.is_ere(node["subject"]) and self.is_ere(node["object"]):
                dot.edge(self.shorten_label(node["subject"]), self.shorten_label(node["object"]), label = node["predicate"])

            else:
                if unary_stmt:
                    # include unary statements too
                    if self.is_ere(node["subject"]):
                        dot.edge(self.shorten_label(node["subject"]), self.shorten_label(node["subject"]), \
                                     label = self.shorten_label(node["object"]))
        

        if outfilename is not None:
            dot.render(outfilename, view=showview)
        else:
            dot.render(view=showview)

        return dot
    
    ###############################

    ###
    # retain only names that are probably English
    def english_names(self, labellist):
        return [label for label in labellist if re.search(r"^[A-Za-z0-9\-,\.\'\"\(\)\? ]+$", label)]

    ###
    # given a label, shorten it for easier reading
    def shorten_label(self, label):
        return label.split("/")[-1].split("#")[-1]

    
    ##
    # string constants in this graph: they always appear as objects in statements
    def _each_string_constant_of_graph(self):
        for stmtlabel, stmtnode in self.each_statement():
            if stmtnode["object"] not in self.thegraph:
                yield stmtnode["object"]
                yield self.shorten_label(stmtnode["object"])

    #####################################3
    # ontology mapping issues
    ##
    # role labels of affiliates in an affiliation
    def is_affiliate_rolelabel(self, label):
        if not label.startswith("GeneralAffiliation") and not label.startswith("OrganizationAffiliation"):
            return False

        rolelabels = [ "Affiliate", "MORE_Person", "Sponsorship_Entity",
                           "EmploymentMembership_Employee", "Founder_Founder",
                           "InvestorShareholder_InvestorShareholder", "ControlTerritory_Controller",
                           "NationalityCitizen_Artifact", "OwnershipPossession_Artifact",
                           "ArtifactPoliticalOrganizationReligiousAffiliation_Artifact",
                           "Ethnicity_Person", "NationalityCitizen_Citizen",
                           "MemberOriginReligionEthnicity_Person", "NationalityCitizen_Organization", 
                           "OrganizationPoliticalReligiousAffiliation_Organization",
                           "OrganizationWebsite_Organization", "AdvisePlanOrganize_ActorOrEvent", 
                           "Affiliated_ActorOrEvent", "HelpSupport_ActorOrEvent", "Sponsorship_ActorOrEvent",
                           "Leadership_Leader", "Ownership_Organization", "StudentAlum_StudentAlum"]

        if any(label.endswith(r) for r in rolelabels):
            return True

        return False

    def is_affiliation_rolelabel(self, label):
        if not label.startswith("GeneralAffiliation") and not label.startswith("OrganizationAffiliation"):
            return False

        rolelabels = [ "Affiliation", "OPRA_Organization", "Sponsorship_Sponsor",
                           "EmploymentMembership_Organization", "Founder_Organization",
                           "ControlTerritory_Territory", "NationalityCitizen_Nationality",
                           "OwnershipPossession_Owner",
                           "ArtifactPoliticalOrganizationReligiousAffiliation_EntityOrFiller",
                           "Ethnicity_Ethnicity", "NationalityCitizen_Nationality",
                           "MemberOriginReligionEthnicity_EntityOrFiller",
                           "OrganizationPoliticalReligiousAffiliation_EntityOrFiller",
                           "OrganizationWebsite_Website", "AdvisePlanOrganize_Sponsor",
                           "Affiliated_Sponsor", "HelpSupport_Sponsor", "Sponsorship_Sponsor",
                           "InvestorShareholder_Organization", "Leadership_Organization",
                           "Ownership_Owner", "StudentAlum_Organization"]

        if any(label.endswith(r) for r in rolelabels):
            return True
        
        return False

    def rolelabel_isa(self, label, eventrel_class, rolelabel):
        if label.startswith(eventrel_class) and label.endswith(rolelabel): return True
