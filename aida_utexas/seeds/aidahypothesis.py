# Katrin Erk April 2019
# Simple class for keeping an AIDA hypothesis.
# As a data structure, a hypothesis is simply a list of statements.
# We additionally track core statements so they can be visualized separately if needed.
# The class also provides access functions to determine EREs that are mentioned in the hypothesis,
# as well as their types, arguments, etc.
# All this is computed dynamically and not kept in the data structure. 


########
# one AIDA hypothesis
from collections import defaultdict


class AidaHypothesis:
    def __init__(self, graph_obj, stmts = None, stmt_weights = None, core_stmts = None, lweight = 0.0):
        self.graph_obj = graph_obj
        if stmts is None:
            self.stmts = set()
        else:
            self.stmts = set(stmts)

        if stmt_weights is None:
            self.stmt_weights = { }
        else:
            self.stmt_weights = stmt_weights

        if core_stmts is None:
            self.core_stmts = set()
        else:
            self.core_stmts = set(core_stmts)

        # failed queries are added from outside, as they are needed in the json object
        self.failed_queries = [ ]

        # hypothesis weight
        self.lweight = lweight

        # query variables and fillers, for quick access to the core answer that this query gives
        self.qvar_filler = { }

        # weight if none is given
        self.default_weight = -100.0

    #####
    # extending a hypothesis: adding a statement outright,
    # or making a new hypothesis and adding the statement there
    def add_stmt(self, stmtlabel, core = False, weight = None):
        if stmtlabel in self.stmts:
            return
        
        if core:
            self.core_stmts.add(stmtlabel)

        self.stmts.add(stmtlabel)

        if weight is not None:
            self.stmt_weights[ stmtlabel ] = weight
        elif core:
            self.stmt_weights[stmtlabel ] = 0.0
        else:
            self.stmt_weights[stmtlabel] = self.default_weight

    def extend(self, stmtlabel, core = False, weight = None):
        if stmtlabel in self.stmts:
            return self
        
        new_hypothesis = self.copy()
        new_hypothesis.add_stmt(stmtlabel, core = core, weight = weight)
        return new_hypothesis

    def copy(self):
        new_hypothesis = AidaHypothesis(
            self.graph_obj,
            stmts = self.stmts.copy(),
            core_stmts = self.core_stmts.copy(),
            stmt_weights = self.stmt_weights.copy(),
            lweight=self.lweight
        )
        new_hypothesis.add_failed_queries(self.failed_queries)
        return new_hypothesis

    #########3
    # give weights to type statements in this hypothesis:
    # single type, or type of an event/relation used in an event/relation argument:
    # weight of maximum neighbor.
    # otherwise, low default weight
    def set_typeweights(self):

        # map each ERE to adjacent type statements
        ere_types = { }
        for stmt in self.stmts:
            if self.graph_obj.is_typestmt(stmt):
                ere = self.graph_obj.stmt_subject(stmt)
                if ere is not None:
                    if ere not in ere_types: ere_types[ere] = [ ]
                    ere_types[ere].append(stmt)

        for ere, typestmts in ere_types.items():
            ere_weight = max(self.stmt_weights.get(stmt, self.default_weight) for stmt in self.graph_obj.each_ere_adjacent_stmt_anyrel(ere))
            # print("HIER1", ere, ere_weight)
            if len(typestmts) == 1:
                # one type statement only: gets weight of maximum-weight edge adjacent to ere
                # print("HIER2 single type statement", ere, typestmts[0], ere_weight)
                self.stmt_weights[ typestmts[0] ] = ere_weight
            else:
                # multiple type statements for ere
                # what outgoing event/relation edges does it have?
                eventrel_roles_ere = [ self.graph_obj.shorten_label(rolelabel) for rolelabel, arg in self.eventrelation_each_argument(ere)]
                for typestmt in typestmts:
                    typelabel = self.graph_obj.stmt_object(typestmt)
                    if typelabel is None:
                        self.stmt_weights[stmt] = self.default_weight
                        continue
                    
                    typelabel = self.graph_obj.shorten_label(typelabel)
                    if any(rolelabel.startswith(typelabel) and len(rolelabel) > len(typelabel) for rolelabel in eventrel_roles_ere):
                        # this is a type that is used in an outgoing edge of this ERE
                        # print("HIER3 used type", ere, typelabel, eventrel_roles_ere, ere_weight)
                        self.stmt_weights[ typestmt ] = ere_weight
                    else:
                        # no reason not to give a low default weight to this edge
                        # print("HIER4 default wt", ere, typelabel, self.default_weight)
                        self.stmt_weights[stmt] = self.default_weight
                        
        
    ########
    # json output as a hypothesis
    # make a json object describing this hypothesis
    def to_json(self):
        stmtlist = list(self.stmts)
        return {
            "statements" : stmtlist,
            "statementWeights" : [ self.stmt_weights.get(s, self.default_weight) for s in stmtlist],
            "failedQueries": self.failed_queries,
            "queryStatements" : list(self.core_stmts)
            }

    def add_failed_queries(self, failed_queries):
        self.failed_queries = failed_queries

    # update the log weight 
    def update_lweight(self, added_lweight):
        self.lweight += added_lweight

    def add_qvar_filler(self, qvar_filler):
        self.qvar_filler = qvar_filler
    
    ########
    # readable output: return EREs in this hypothesis, and the statements associated with them
    # string for single ERE
    def ere_to_s(self, ere_id, roles_ontology):
        if self.graph_obj.is_event(ere_id) or self.graph_obj.is_relation(ere_id):
            return self.eventrel_to_s(ere_id, roles_ontology)
        elif self.graph_obj.is_entity(ere_id):
            return self.entity_to_s(ere_id)
        else:
            return ""

    def entity_to_s(self, ere_id):
        retv = self.entity_best_name(ere_id)
        if retv == '[unknown]':
            for typelabel in self.ere_each_type(ere_id):
                retv = typelabel
                break
        return retv

    def eventrel_to_s(self, ere_id, roles_ontology):
        if not (self.graph_obj.is_event(ere_id) or self.graph_obj.is_relation(ere_id)):
            return ''

        eventrel_type = None
        for typelabel in self.ere_each_type(ere_id):
            eventrel_type = typelabel
            break

        arg_values = defaultdict(set)
        for arglabel, value in self.eventrelation_each_argument(ere_id):
            arg_values[arglabel].add(value)

        if not arg_values:
            return ''

        if eventrel_type is None:
            eventrel_type = list(arg_values.keys())[0].rsplit('_', maxsplit=1)[0]

        retv = eventrel_type
        for arg_key in roles_ontology[eventrel_type].values():
            retv += '\n    ' + arg_key + ': '
            arglabel = eventrel_type + '_' + arg_key
            if arglabel in arg_values:
                retv += ', '.join(self.entity_to_s(arg_id) for arg_id in arg_values[arglabel])

        # for arglabel, values in arg_values.items():
        #     retv += '\n' + '    ' + arglabel.rsplit('_', maxsplit=1)[1]
        #     retv += ': ' + ', '.join(self.entity_to_s(arg_id) for arg_id in values)

        return retv

    def _entity_to_s(self, ere_id, prefix = ""):
        retv = ""
        # print info on this argument
        retv += prefix + self.nodetype(ere_id) + " " + ere_id + "\n"

        # add type information if present
        for typelabel in self.ere_each_type(ere_id):
            retv += prefix + "ISA: " + typelabel + "\n"

        # print string label
        name = self.entity_best_name(ere_id)
        retv += prefix + "handle: " + name + "\n"

        return retv

    def _eventrel_to_s(self, ere_id, prefix = "", withargs = True):
        retv = ""
        if not (self.graph_obj.is_event(ere_id) or self.graph_obj.is_relation(ere_id)):
            return retv

        retv += prefix + self.nodetype(ere_id) + " " + ere_id + "\n"

        # add type information if present
        for typelabel in self.ere_each_type(ere_id):
            retv += prefix + "ISA: " + typelabel + "\n"

        # add argument information:
        if withargs:
            # first sort by argument label
            arg_values = { }
            for arglabel, value in self.eventrelation_each_argument(ere_id):
                if arglabel not in arg_values: arg_values[arglabel] = set()
                arg_values[ arglabel ].add(value)

            # then add to string
            for arglabel, values in arg_values.items():
                retv += "\n" + prefix + "  " + arglabel + "\n"
                additionalprefix = "    "
                for arg_id in values:
                    retv += prefix + self._entity_to_s(arg_id, prefix + additionalprefix)

        return retv

    # String for whole hypothesis
    def to_s(self, roles_ontology):
        retv = ""

        # retv += ", ".join(sorted(self.stmts, key = lambda s:self.stmt_weights[s], reverse = True)) + "\n\n"

        # start with core statements
        core = self.core_eres()
        for ere_id in core:
            if self.graph_obj.is_event(ere_id) or self.graph_obj.is_relation(ere_id):
                ere_str = self.ere_to_s(ere_id, roles_ontology)
                if ere_str != '':
                    retv += ere_str + "\n\n"

        # make output for each event or relation in the hypothesis
        for ere_id in self.eres():
            if ere_id in core:
                # already done
                continue

            if self.graph_obj.is_event(ere_id):
                ere_str = self.ere_to_s(ere_id, roles_ontology)
                if ere_str != '':
                    retv += ere_str + "\n\n"

        # make output for each event or relation in the hypothesis
        for ere_id in self.eres():
            if ere_id in core:
                # already done
                continue

            if self.graph_obj.is_event(ere_id):
                ere_str = self.ere_to_s(ere_id, roles_ontology)
                if ere_str != '':
                    retv += ere_str + "\n\n"

        return retv

    # String for a statement
    def statement_to_s(self, stmtlabel):
        if stmtlabel not in self.stmts:
            return ""

        retv = ""
        stmt = self.graph_obj.thegraph[stmtlabel]
        retv += "Statement " + stmtlabel + "\n"
        
        retv += "Subject:\n " 
        if self.graph_obj.is_node(stmt["subject"]):
            retv += self.ere_to_s(stmt["subject"], withargs = False, prefix = "    ") + "\n"
        else:
            retv += stmt["subject"] + "\n"

        retv += "Predicate: " + stmt["predicate"] + "\n"

        retv += "Object:\n " 
        if self.graph_obj.is_node(stmt["object"]):
            retv += self.ere_to_s(stmt["object"], withargs = False, prefix = "    ") + "\n"
        else:
            retv += stmt["object"] + "\n"
        

    #############
    # access functions

    # list of EREs adjacent to the statements in this hypothesis
    def eres(self):
        return list(set(nodelabel for stmtlabel in self.stmts for nodelabel in self.graph_obj.statement_args(stmtlabel) \
                        if self.graph_obj.is_ere(nodelabel)))

    # list of EREs adjacent to core statements of this hypothesis
    def core_eres(self):
        return list(set(nodelabel for stmtlabel in self.core_stmts for nodelabel in self.graph_obj.statement_args(stmtlabel) \
                        if self.graph_obj.is_ere(nodelabel)))
                        
    def eres_of_stmt(self, stmtlabel):
        if stmtlabel not in self.stmts:
            return [ ]
        else:
            return list(set(nodelabel for nodelabel in self.graph_obj.statement_args(stmtlabel) \
                        if self.graph_obj.is_ere(nodelabel)))


    # iterate over arguments of an event or relation in this hypothesis
    # yield tuples of (statement, argument label, ERE ID)
    def eventrelation_each_argstmt(self, eventrel_id):
        if not (self.graph_obj.is_event(eventrel_id) or self.graph_obj.is_relation(eventrel_id)):
            return

        for stmtlabel in self.graph_obj.each_ere_adjacent_stmt_anyrel(eventrel_id):
            if stmtlabel in self.stmts:
                stmt = self.graph_obj.thegraph[stmtlabel]
                if stmt["subject"] == eventrel_id and self.graph_obj.is_ere(stmt["object"]):
                    yield (stmtlabel, stmt["predicate"], stmt["object"])

    # iterate over arguments of an event or relation in this hypothesis
    # yield pairs of (argument label, ERE ID)
    def eventrelation_each_argument(self, eventrel_id):
        for stmtlabel, predicate, object in self.eventrelation_each_argstmt(eventrel_id):
            yield (predicate, object)

    # return each argument of the event or relation eventrel_id that has rolelabel as its label
    # exact match!!
    def eventrelation_each_argument_labeled(self, eventrel_id, rolelabel):
        for thisrolelabel, filler in self.eventrelation_each_argument(eventrel_id):
            if thisrolelabel == rolelabel:
                yield filler

    # return each argument of the event or relation eventrel_id that has rolelabel as its label
    # exact match!!
    def eventrelation_each_argument_labeled_like(self, eventrel_id, classlabel, rolelabel):
        for thisrolelabel, filler in self.eventrelation_each_argument(eventrel_id):
            if self.graph_obj.rolelabel_isa(thisrolelabel, classlabel, rolelabel):
                yield filler
                
    def eventrelation_each_argstmt_labeled(self, eventrel_id, rolelabel):
        for stmt, thisrolelabel, filler in self.eventrelation_each_argstmt(eventrel_id):
            if thisrolelabel == rolelabel:
                yield (stmt, filler)
                
    def eventrelation_each_argstmt_labeled_like(self, eventrel_id, classlabel, rolelabel):
        for stmt, thisrolelabel, filler in self.eventrelation_each_argstmt(eventrel_id):
            if self.graph_obj.rolelabel_isa(thisrolelabel, classlabel, rolelabel):
                yield (stmt, filler)
        
            

    # types of an ERE node in this hypothesis
    def ere_each_type(self, ere_id):
        if not self.graph_obj.is_ere(ere_id):
            return
        for stmtlabel in self.graph_obj.each_ere_adjacent_stmt(ere_id, "type", "subject"):
            if stmtlabel in self.stmts:
                yield self.graph_obj.shorten_label(self.graph_obj.thegraph[stmtlabel]["object"])
            
        
    
    # node type: Entity, Event, Relation, Statement
    def nodetype(self, nodelabel):
        if self.graph_obj.is_node(nodelabel):
            return self.graph_obj.thegraph[nodelabel]["type"]
        else:
            return None

    # names of an entity
    def entity_names(self, ere_id):
        #return self.graph_obj.english_names(self.graph_obj.ere_names(ere_id))
        return self.graph_obj.ere_names(ere_id)

    # "best" name of an entity
    def entity_best_name(self, ere_id):
        names = self.entity_names(ere_id)
        if names is None or names == [ ]:
            return "[unknown]"
        english_names = self.graph_obj.english_names(names)
        if len(english_names) > 0:
            return min(english_names, key = lambda n:len(n))
        else:
            return min(names, key = lambda n: len(n))


    # possible affiliations of an ERE:
    # yield ERE that is the affiliation
    def ere_each_possible_affiliation(self, ere_id):
        for affiliation_id in self.graph_obj.possible_affiliations(ere_id):
            yield affiliation_id

    # actual affiliations of an ERE in this hypothesis
    def ere_each_affiliation(self, ere_id):
        for stmt1, affiliation_ere, stmt2 in self.graph_obj.possible_affiliation_triples(ere_id):
            if stmt1 in self.stmts and stmt2 in self.stmts:
                affiliation_id = self.graph_obj.stmt_object(stmt2)
                yield affiliation_id

    # is this ERE listed as an affiliation, though not necessarily in this hypothesis?
    def ere_possibly_isaffiliation(self, ere_id):
        for stmt in self.graph_obj.each_ere_adjacent_stmt_anyrel(ere_id):
            if self.graph_obj.stmt_object(stmt) == ere_id and self.graph_obj.is_affiliation_rolelabel(self.graph_obj.stmt_predicate(stmt)):
                return True
        
############3
# collection of hypotheses, after initial cluster seed generation has been done
class AidaHypothesisCollection:
    def __init__(self, hypotheses):
        self.hypotheses = hypotheses

    def add(self, hypothesis):
        self.hypotheses.append(hypothesis)
        
    # compile json object that lists all the hypotheses with their statements
    def to_json(self):
       
        # make a json in the right format.
        # entries: "probs", "support". "probs": add dummy uniform probabilities
        json_out = { "probs": [h.lweight for h in self.hypotheses],
                     "support" : [ ]
                   }
        for hyp in self.hypotheses:
            json_out["support"].append(hyp.to_json())

        return json_out

    # make a list of strings with the new cluster seeds in readable form
    def to_s(self):
        return [ hyp.to_s() for hyp in self.hypotheses ]
        
    @staticmethod
    # generate an AidaHypothesisCollection from a json file
    def from_json(json_obj, graph_obj):
        def hypothesis_from_json(j, wt):
            h = AidaHypothesis(graph_obj, stmts = j["statements"], core_stmts = j["queryStatements"],
                                    stmt_weights = dict((j["statements"][i], j["statementWeights"][i]) for i in range(len(j["statements"]))),
                                   lweight = wt)
        
            h.add_failed_queries(j["failedQueries"])
            return h
            
        return AidaHypothesisCollection([hypothesis_from_json(json_obj["support"][i], json_obj["probs"][i]) for i in range(len(json_obj["support"]))])

        
        

