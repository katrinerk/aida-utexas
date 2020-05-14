# Katrin Erk March 2019
# Rule-based creation of initial hypotheses
# This only add the statements that the Statement of Information Need asks for,
# but constructs all possible cluster seeds that can be made using different statements
# that all fill the same SOIN


import sys
from collections import deque, defaultdict
import math
import itertools
from operator import itemgetter

from aida_utexas.seeds.aidahypothesis import AidaHypothesis, AidaHypothesisCollection
from aida_utexas.seeds.hypothesisfilter import AidaHypothesisFilter
from aida_utexas.seeds.datecheck import AidaIncompleteDate, temporal_constraint_match


#########
# class that holds a single cluster seed.
# just a data structure, doesn't do much.
class OneClusterSeed:
    def __init__(self, graph_obj, core_constraints, temporal_constraints, hypothesis, qvar_filler, lweight = 0.0,
                    unfilled = None, unfillable = None, entrypoints = None, entrypointweight = 0.0):
        # the following data is not changed, and is kept just for info
        self.graph_obj = graph_obj
        self.core_constraints = core_constraints
        # temporal constraints: mapping queryvariable -> {start_time: ..., end_time:...}
        self.temporal_constraints = temporal_constraints
        
        # the following data is changed.
        # flags: am I done?
        self.done = False

        # what is my current log weight?
        self.lweight = lweight
        # weight according to the entry points
        self.entrypointweight = entrypointweight
        # print("HIEr entrypwt", self.entrypointweight)
        # input()
        
        # hypothesis is an AidaHypothesis object
        self.hypothesis = hypothesis

        # hypothesis filter
        self.filter = AidaHypothesisFilter(self.graph_obj)

        # mapping from query variable to filler: string to string, value strings are IDs in graph_obj
        self.qvar_filler = qvar_filler
        # entry points (will not be used in ranking this hypothesis later)
        self.entrypoints = entrypoints

        # unfilled, unfillable are indices on self.core_constraints
        if unfilled is None: self.unfilled = set(range(len(core_constraints)))
        else: self.unfilled = unfilled

        if unfillable is None: self.unfillable = set()
        else: self.unfillable = unfillable

        # some weights for things that might go wrong during query creation
        self.FAILED_QUERY_WT = -0.1
        self.FAILED_TEMPORAL = -0.1
        self.FAILED_ONTOLOGY = -0.1
        self.DUPLICATE_FILLER = -0.01


    # finalize:
    # report failed queries ot the underlying AidaHypothesis object
    def finalize(self):

        self.hypothesis.add_failed_queries( list(map( lambda ix: self.core_constraints[ix], self.unfillable)) )
        self.hypothesis.update_lweight(self.lweight)
        self.hypothesis.add_qvar_filler(self.qvar_filler)

        return self.hypothesis
            
    # extend hypothesis by one statement filling the next fillable core constraint.
    # returns a list of OneClusterSeed objects
    def extend(self):
        nfc = self._next_fillable_constraint()

        if nfc is None:
            # no next fillable constraint.
            # declare done, and return only this object
            self.done = True
            # no change to qvar_filler
            self.unfillable.update(self.unfilled)
            self.unfilled = set()
            return [ self ]

        elif nfc["failed"]:
            # this particular constraint was not fillable, and will never be fillable.
            # print("failed constraint", self.core_constraints[nfc["index"]][1])# , [(q, v[-5:]) for q, v in self.qvar_filler.items()])
            self.unfilled.remove(nfc["index"])
            self.unfillable.add(nfc["index"])
            # update the weight
            # print("adding failed query weight", self.lweight, self.lweight + self.FAILED_QUERY_WT)
            self.lweight += self.FAILED_QUERY_WT
            return [self ]
            
        else:
            # nfc is a structure with entries "predicate", "role", "erelabel", "variable"
            # find statements that match this constraint, and
            # return a list of extended hypotheses to match this.
            # these hypotheses have not been run through the filter yet
            # format: list of tuples (new hypothesis, new_stmt, variable, filler)
            new_hypotheses = self._extend(nfc)

            if len(new_hypotheses) == 0:
                # print("HIER no new hypotheses")
                # something has gone wrong
                self.unfilled.remove(nfc["index"])
                self.unfillable.add(nfc["index"])
                # update the weight
                # print("adding failed query weight", self.lweight, self.lweight + self.FAILED_QUERY_WT)
                self.lweight += self.FAILED_QUERY_WT
                return [self ]

            # determine the constraint that we are matching
            # and remove it from the list of unfilled constraints
            constraint = self.core_constraints[ nfc["index"] ]

            retv = [ ]
            for new_hypothesis, stmtlabel, variable, filler in new_hypotheses:
                add_weight = 0
                
                if self.filter.validate(new_hypothesis, stmtlabel):
                    # yes: make a new OneClusterSeed object with this extended hypothesis
                    new_qvar_filler = self.qvar_filler.copy()
                    if variable is not None and filler is not None and self.graph_obj.is_ere(filler):
                        new_qvar_filler[ variable] = filler
                        if filler in self.qvar_filler.values():
                            # some other variable has been mapped to the same ERE
                            add_weight += self.DUPLICATE_FILLER
                            # print("duplicate filler", new_qvar_filler)
                            

                    # changes to unfilled, not to unfillable
                    new_unfilled = self.unfilled.difference([nfc["index"]])
                    new_unfillable = self.unfillable.copy()

                    if nfc["relaxed"]:
                        # print("adding failed ontology weight", self.lweight, self.lweight + self.FAILED_ONTOLOGY)
                        add_weight += self.FAILED_ONTOLOGY

                    retv.append(OneClusterSeed(self.graph_obj, self.core_constraints, self.temporal_constraints, new_hypothesis, new_qvar_filler,
                                                lweight = self.lweight + add_weight, entrypointweight = self.entrypointweight,
                                                unfilled = new_unfilled, unfillable = new_unfillable,
                                                entrypoints = self.entrypoints))

                ## else:
                ##     print("failed to validate", stmtlabel, self.graph_obj.stmt_predicate(stmtlabel))

            if len(retv) == 0:
                # all the fillers were filtered away
                self.unfilled.remove(nfc["index"])
                self.unfillable.add(nfc["index"])
                # update the weight
                # print("all candidate statements filtered away, adding failed query weight", self.lweight, self.lweight + self.FAILED_QUERY_WT)
                self.lweight += self.FAILED_QUERY_WT
                return  [ self ]
            else:
                return retv
        
    # return true if there is at least one unfilled core constraint remaining
    def core_constraints_remaining(self):
        return len(self.unfilled) > 0

    # return true if there are no unfillable constraints
    def no_failed_core_constraints(self):
        return len(self.unfillable) == 0

    def has_statements(self):
        return len(self.hypothesis.stmts) > 0
    
    # next fillable constraint from the core constraints list,
    # or None if none fillable
    def _next_fillable_constraint(self):
        # iterate over unfilled query constraints to see if we can find one that can be filled
        for constraint_index in self.unfilled:
            
            subj, pred, obj = self.core_constraints[constraint_index]
            # print("HIER", subj, pred, obj, self.qvar_filler)

            # if either subj or obj is known (is an ERE or has an entry in qvar_filler,
            # then we should be able to fill this constraint now, or it is unfillable
            subj_filler = self._known_coreconstraintentry(subj)
            obj_filler = self._known_coreconstraintentry(obj)

            if subj_filler is not None and obj_filler is not None:
                # new edge between two known variables
                return self._fill_constraint_knowneres(constraint_index, subj_filler, pred, obj_filler)
            
            elif subj_filler is not None:
                return self._fill_constraint(constraint_index, subj_filler, "subject", pred, obj, "object")

            elif obj_filler is not None:
                return self._fill_constraint(constraint_index, obj_filler, "object", pred, subj, "subject")
                
            else:
                # this constraint cannot be filled at this point,
                # wait and see if it can be filled some other time
                continue

        # reaching this point, and not having returned anything:
        # this means we do not have any fillable constraints left
        return None

            



    # given a subject or object from a core constraint, is this an ERE ID from the graph
    # or a variable for which we already know the filler?
    # if so, return the filler ERE ID. otherwise none
    def _known_coreconstraintentry(self, entry):
        if entry in self.graph_obj.thegraph: return entry
        elif entry in self.qvar_filler: return self.qvar_filler[entry]
        else: return None

    # see if this label can be generalized by cutting out the lowest level of specificity.
    # returns: generalized label, plus role (or None)
    def _generalize_label(self, label):
        pieces = label.split("_")
        if len(pieces) == 1:
            labelclass = label
            labelrole = ""
        elif len(pieces) == 2:
            labelclass = pieces[0]
            labelrole = pieces[1]
        else:
            print("unexpected number of underscores in label, could not split", label)
            return (None, None)
    
        pieces = labelclass.split(".")
        if len(pieces) <= 2:
            # no more general class possible
            return (None, None)
        
        # we can try a more lenient match
        labelclass = ".".join(pieces[:-1])
        return (labelclass, labelrole)

    # returns list of statement candidates bordering ERE
    # that have ERE in role 'role' (subject, object) and have predicate 'pred'.
    # also returns whether the statements had to be relaxed.
    # If no statements could be found, returnsNone
    def _statement_candidates(self, ere, pred, role):
        candidates = list(self.graph_obj.each_ere_adjacent_stmt(ere, pred, role))

        if len(candidates) > 0:
            # success, we found some
            return { "candidates" : candidates,
                     "relaxed" : False }

        # no success. see if more lenient match will work
        lenient_pred, lenient_role = self._generalize_label(pred)

        # print("no match for", pred, "checking", lenient_pred, lenient_role)
        
        if lenient_pred is None:
            # no generalization possible
            return None
        else:
            # try the more general class
            candidates = []
            for stmt in self.graph_obj.each_ere_adjacent_stmt_anyrel(ere):
                if self.graph_obj.stmt_predicate(stmt).startswith(lenient_pred) and self.graph_obj.stmt_predicate(stmt).endswith(lenient_role):
                    candidates.append(stmt)

            if len(candidates) > 0:
                # success, we found some
                # print("success")
                return { "candidates" : candidates,
                            "relaxed" : True }
            else:
                return None
                
    # try to fill this constraint from the graph, either strictly or leniently.
    # one side of this constraint is a known ERE, the other side can be anything
    def _fill_constraint(self, constraint_index, knownere, knownrole, pred, unknown, unknownrole):
        # find statements that could fill the role
        candidates = self._statement_candidates(knownere, pred, knownrole)
        
        if candidates is None:
            # no candidates found at all, constraint is unfillable
            return { "index" : constraint_index,
                    "failed" : True
                         }
        
        # check if unknown is a constant in the graph,
        # in which case it is not really unknown
        if self._is_string_constant(unknown):
            # which of the statement candidates have the right filler?
            candidates = [s for s in candidates if self.graph_obj.thegraph[s][unknownrole] == unknown]
            if len(candidates) == 0:
                return { "index" : constraint_index,
                    "failed" : True
                         }
            else:
                return {
                    "index" : constraint_index,
                    "stmt" : candidates["candidates"],
                    "role" : knownrole,
                    "has_variable" : False,
                    "relaxed" : candidates["relaxed"],
                    "failed" : False
                    }
                        
        else:
            # nope, we have a variable we can fill
            # any fillers?
            return {
                "index" : constraint_index,
                "stmt" : candidates["candidates"],
                "variable" : unknown,
                "role" : knownrole,
                "has_variable" : True,
                "relaxed" : candidates["relaxed"],
                "failed" : False
                }
        
    # try to fill this constraint from the graph, either strictly or leniently.
    # both sides of this constraint are known EREs
    def _fill_constraint_knowneres(self, constraint_index, ere1, pred, ere2):
        # find statements that could fill the role
        possible_candidates = self._statement_candidates(ere1, pred, "subject")

        if possible_candidates is None:
            # no candidates found at all, constraint is unfillable
            return { "index" : constraint_index,
                    "failed" : True
                         }

        # we did find candidates.
        # check whether any of the candidates has ere2 as its object
        candidates = [c for c in possible_candidates["candidates"] if self.graph_obj.stmt_object == ere2]
        if len(candidates) == 0:
            # constraint is unfillable
            return { "index" : constraint_index,
                    "failed" : True
                         }

        else:
            return { "index" : constraint_index,
                         "failed" : False,
                         "stmt" : candidates,
                         "has_variable" : False,
                         "relaxed" : candidates["relaxed"]
                         }
            
        

    # nfc is a structure with entries "predicate", "role", "erelabel", "variable"
    # find statements that match this constraint, and
    # return a list of triples; (extended hypothesis, query_variable, filler)
    def _extend(self, nfc):

        if len(nfc["stmt"]) == 0:
            # did not find any matches to this constraint
            return [ ]

        if not nfc["has_variable"]:
            # this next fillable constraint states a constant string value about a known ERE,
            # or it states a new connection between known EREs.
            # we do have more than zero matching statements. add just the first one, they are identical
            stmtlabel = nfc["stmt"][0]
            if stmtlabel not in self.graph_obj.thegraph:
                print("Error in ClusterSeed: unexpectedly did not find statement", stmtlabel)
                return [ ]
                
            else:
                # can this statement be added to the hypothesis without contradiction?
                # extended hypothesis
                return [ (self.hypothesis.extend(stmtlabel, core = True), stmtlabel, None, None)]

        # we know that we have a variable now.
        # it is in nfc["variable"]
        # determine the role that the variable is filling, if there is a variable
        # make a new hypothesis for each statement that could fill the current constraint.
        # if we don't find anything, re-run with more leeway on temporal constraints
        retv, has_temporal_constraint = self._extend_withvariable(nfc, 0)
        if len(retv) == 0 and has_temporal_constraint:
            # relax temporal matching by one day
            # update weight to reflect relaxing of temporal constraint
            # print("adding failed temporal weight", self.lweight, self.lweight + self.FAILED_TEMPORAL)
            self.lweight += self.FAILED_TEMPORAL
            retv, has_temporal_constraint = self._extend_withvariable(nfc, 1)
            
        if len(retv) == 0 and has_temporal_constraint:
            # relax temporal matching: everything goes
            # print("adding failed temporal weight", self.lweight, self.lweight + self.FAILED_TEMPORAL)
            self.lweight += self.FAILED_TEMPORAL
            retv, has_temporal_constraint = self._extend_withvariable(nfc, 2)
        
        return retv

    # nfc is a structure with entries "predicate", "role", "erelabel", "variable"
    # find statements that match this constraint, and
    # return a pair (hyp, has_temporal_cosntraint) where
    # hyp is a list of triples; (extended hypothesis, query_variable, filler)
    # and has_temporal_constraint is true if there was at least one temporal constraint that didn't get matched
    def _extend_withvariable(self, nfc, leeway = 0):

        retv = [ ]
        has_temporal_constraint = False
        
        otherrole = self._nfc_otherrole(nfc)
        if otherrole is None:
            # some error
            return (retv, has_temporal_constraint)

        
        for stmtlabel in nfc["stmt"]:
            if stmtlabel not in self.graph_obj.thegraph:
                print("Error in ClusterSeed: unexpectedly did not find statement", stmtlabel)
                continue

            # determine the entity or value that fills the role that has the variable
            filler = self.graph_obj.thegraph[stmtlabel][otherrole]

            # is this an entity? if so, we need to check for temporal constraints.
            if filler in self.graph_obj.thegraph:
                # is there a problem with a temporal constraint?
                if not temporal_constraint_match(self.graph_obj.thegraph[filler], self.temporal_constraints.get(nfc["variable"], None), leeway):
                    # yup, this filler runs afoul of some temporal constraint.
                    # do not use it
                    # print("temp mismatch")
                    has_temporal_constraint = True
                    continue

                # we also check wether including this statement will violate another constraint.
                # if so, we do  not include it
                if self._second_constraint_violated(nfc["variable"], filler, nfc["index"]):
                    # print("second constraint violated, skipping", stmtlabel, self.graph_obj.stmt_predicate(stmtlabel))
                    continue

            # can this statement be added to the hypothesis without contradiction?
            # extended hypothesis
            new_hypothesis = self.hypothesis.extend(stmtlabel, core = True)
            retv.append( (new_hypothesis, stmtlabel, nfc["variable"], filler) )

        return (retv, has_temporal_constraint)

    # second constraint violated: given a variable and its filler,
    # see if filling this qvar with this filler will make any constraint that is yet unfilled unfillable
    def _second_constraint_violated(self, variable, filler, exceptindex):
        for constraint_index in self.unfilled:
            if constraint_index == exceptindex:
                # this was the constraint we were just going to fill, don't re-check it
                continue
            
            subj, pred, obj = self.core_constraints[constraint_index]
            if subj == variable and obj in self.qvar_filler:
                # found a constraint involving this variable and another variable that has already been filled
                candidates = self._statement_candidates(filler, pred, "subject")
                if candidates is None:
                    ## print("trying to add", filler[-5:], "for", variable, "originally doing", self.core_constraints[exceptindex], exceptindex, constraint_index)
                    ## print("could not fill", subj, pred, obj, "where obj is ", self.qvar_filler[obj][-5:])
                    ## input()
                    return True
                else:
                    candidates = [c for c in candidates["candidates"] if self.graph_obj.stmt_object == self.qvar_filler[obj]]
                    if len(candidates) == 0:
                        ## print("trying to add", filler[-5:], "for", variable, "originally doing", self.core_constraints[exceptindex], exceptindex, constraint_index)
                        ## print("could not fill", subj, pred, obj, "where obj is ", self.qvar_filler[obj][-5:])
                        ## input()
                        return True
                    
            elif obj == variable and subj in self.qvar_filler:                
                # found a constraint involving this variable and another variable that has already been filled
                candidates = self._statement_candidates(filler, pred, "object")
                if candidates is None:
                    ## print("trying to add", filler[-5:], "for", variable, "originally doing", self.core_constraints[exceptindex], exceptindex, constraint_index)
                    ## print("could not fill", subj, pred, obj, "where subj is ", self.qvar_filler[subj][-5:])
                    ## input()
                    return True
                else:
                    candidates = [c for c in candidates["candidates"] if self.graph_obj.stmt_subject == self.qvar_filler[subj]]
                    if len(candidates) == 0:
                        ## print("trying to add", filler[-5:], "for", variable, "originally doing", self.core_constraints[exceptindex], exceptindex, constraint_index)
                        ## print("could not fill", subj, pred, obj, "where subj is ", self.qvar_filler[subj][-5:])
                        ## input()
                        return True

        return False
                
    # given a next_fillable_constraint dictionary,
    # if it has a role of "subject" return 'object' and vice versa
    def _nfc_otherrole(self, nfc):
        if nfc["role"] == "subject":
            return "object"
        elif nfc["role"] == "object":
            return "subject"
        else:
            print("ClusterSeed error: unknown role", nfc["role"])
            return None

    # is the given string a variable, or should it be viewed as a string constant?
    # use the list of all string constants in the given graph
    def _is_string_constant(self, strval):
        return strval in self.graph_obj.string_constants_of_graph

      
#########################
#########################
# class that manages all cluster seeds
class ClusterSeeds:
    # initialize with an AidaJson object and a statement of information need,
    # which is just a json object
    def __init__(self, graph_obj, soin_obj, discard_failedqueries = False, earlycutoff = None, qs_cutoff = None):
        self.graph_obj = graph_obj
        self.soin_obj = soin_obj

        # discard queries with any failed constraints?
        self.discard_failedqueries = discard_failedqueries
        # cut off after N entry point combinations?
        self.earlycutoff = earlycutoff
        # cut off partially formed hypotheses during creation
        # if there are at least QS_CUTOFF other hypothesis seeds
        # with the same fillers for QS_COUNT_CUTOFF query variables?
        self.QS_COUNT_CUTOFF = 3
        self.QS_CUTOFF = qs_cutoff


        # parameters for ranking
        self.rank_first_k = 100
        self.bonus_for_novelty = -5
        self.consider_next_k_in_reranking = 10000
        self.num_bins = 5
        

        # make seed clusters
        self.hypotheses = self._make_seeds()

    # export hypotheses to AidaHypothesisCollection
    def finalize(self):

        # ranking is a list of the hypotheses in self.hypotheses,
        # best first
        print("Making the ranking")
        ranking = self._rank_seeds()

        # turn ranking into log weights:
        # meaningless numbers. just assign 1/2, 1/3, 1/4, ...
        for rank, hyp in enumerate(ranking):
            hyp.lweight = math.log(1.0 / (rank + 1))
        
        hypotheses_for_export = [ h.finalize() for h in ranking ] #sorted(self.hypotheses, key = lambda h:h.hypothesis.lweight, reverse = True)]
        return AidaHypothesisCollection( hypotheses_for_export)

    #########################
    #########################
    # HYPOTHESIS SEED CREATION

    # create initial cluster seeds.
    # this is called from __init__
    def _make_seeds(self):
        # keep queue of hypotheses-in-making, list of finished hypotheses
        hypotheses_todo = deque()
        hypotheses_done = [ ]

        # have we found any hypothesis without failed queries yet?
        # if so, we can eliminate all hypotheses with failed queries
        previously_found_hypothesis_without_failed_queries = False

        if self.earlycutoff is not None:
            facet_cutoff = self.earlycutoff / len(self.soin_obj["facets"])


        ## # TESTING
        ## for epvar, epfillers in self.soin_obj["entrypoints"].items():
        ##     for index, epfiller in enumerate(epfillers):
        ##         print(epvar, epfiller, self.soin_obj["entrypointWeights"][epvar][index], self._entrypoint_filler_rolescore(epvar, epfiller, self.soin_obj["facets"][0]))

        ## input()
            
        ################
        print("Initializing cluster seeds (if stalled, set earlycutoff)")
        # initialize deque with one core hypothesis per facet
        for facet in self.soin_obj["facets"]:
            reranked_entrypoints = {}
            reranked_entrypoint_weights = {}

            for ep_var, ep_fillers in self.soin_obj['entrypoints'].items():
                ep_weights = self.soin_obj['entrypointWeights'][ep_var]

                print('Entry point: {}'.format(ep_var))

                filler_weight_mapping = {}

                fillers_filtered_both = []
                fillers_filtered_role_score = []

                for ep_filler, ep_weight in zip(ep_fillers, ep_weights):
                    ep_role_score = self._entrypoint_filler_rolescore(ep_var, ep_filler, facet)
                    filler_weight_mapping[ep_filler] = (ep_weight, ep_role_score)
                    if ep_role_score > 0:
                        fillers_filtered_role_score.append(ep_filler)
                        if ep_weight > 50.0:
                            fillers_filtered_both.append(ep_filler)

                if len(fillers_filtered_both) > 0:
                    print('\tKept {} fillers with both SoIN weight > 50 and role score > 0'.format(len(fillers_filtered_both)))
                    reranked_entrypoints[ep_var] = fillers_filtered_both
                    reranked_entrypoint_weights[ep_var] = [
                        filler_weight_mapping[filler][0] * filler_weight_mapping[filler][1]
                        for filler in fillers_filtered_both]
                elif len(fillers_filtered_role_score) > 0:
                    print('\tKept {} fillers with role score > 0'.format(len(fillers_filtered_role_score)))
                    reranked_entrypoints[ep_var] = fillers_filtered_role_score
                    reranked_entrypoint_weights[ep_var] = [
                        filler_weight_mapping[filler][0] * filler_weight_mapping[filler][1]
                        for filler in fillers_filtered_role_score]
                else:
                    print('\tKept all {} fillers (no filler has role score > 0)'.format(len(ep_fillers)))
                    reranked_entrypoints[ep_var] = ep_fillers
                    reranked_entrypoint_weights[ep_var] = ep_weights

            index = 0
            if self.earlycutoff is None:
                for qvar_filler, entrypoint_weight in self._each_entry_point_combination(reranked_entrypoints, reranked_entrypoint_weights, facet):
                    ## print("entry points")
                    ## for q, f in qvar_filler.items():
                    ##     print(q, f[-5:])
                    ## print("====")
                    # index += 1

                    # if self.earlycutoff is not None and index >= facet_cutoff:
                    #     # print("early cutoff on cluster seeds: breaking off at", index)
                    #     break

                    # start a new hypothesis
                    core_hyp = OneClusterSeed(self.graph_obj, facet["queryConstraints"], self._pythonize_datetime(facet.get("temporal", {})), \
                                                  AidaHypothesis(self.graph_obj), qvar_filler, entrypointweight = entrypoint_weight,
                                                  entrypoints = list(qvar_filler.keys()))
                    hypotheses_todo.append(core_hyp)
            else:
                for qvar_filler, entrypoint_weight in self._each_entry_point_combination_w_early_cutoff(
                        reranked_entrypoints, reranked_entrypoint_weights, facet, earlycutoff=self.earlycutoff):
                    # start a new hypothesis
                    core_hyp = OneClusterSeed(self.graph_obj, facet["queryConstraints"], self._pythonize_datetime(facet.get("temporal", {})), \
                                                  AidaHypothesis(self.graph_obj), qvar_filler, entrypointweight = entrypoint_weight,
                                                  entrypoints = list(qvar_filler.keys()))
                    hypotheses_todo.append(core_hyp)

        ################
        print("Extending cluster seeds (if too many, reduce rank_cutoff)")
        printindex = 0
        # signatures of query variables, for qs cutoff
        qvar_signatures = { }
        
        
        # extend all hypotheses in the deque until they are done
        while len(hypotheses_todo) > 0:
            printindex += 1
            if printindex % 1000 == 0:
                print("hypotheses done", len(hypotheses_done))
                
            core_hyp = hypotheses_todo.popleft()

            if self.QS_CUTOFF is not None:
                qs = self._make_qvar_signature(core_hyp)
                if qs is not None:
                    if any(qvar_signatures.get(q1, 0) >= self.QS_CUTOFF for q1 in qs):
                        # do not process this hypothesis further
                        # print("skipping hypothesis", qs)
                        continue
                    else:
                        for q1 in qs:
                            qvar_signatures[ q1] = qvar_signatures.get(q1, 0) + 1

            if self.discard_failedqueries:
                # we are discarding hypotheses with failed queries
                if previously_found_hypothesis_without_failed_queries and not(core_hyp.no_failed_core_constraints()):
                    # don't do anything with this one, discard
                    # It has failed queries, and we have found at least one hypothesis without failed queries
                    # print("discarding hypothesis with failed queries")
                    continue
                
            if core_hyp.done:
                # hypothesis finished.
                # any statements in this one?
                if not core_hyp.has_statements():
                    # if not, don't record it
                    # print("empty hypothesis")
                    continue

                if self.discard_failedqueries and core_hyp.no_failed_core_constraints():
                    # yes, no failed queries!
                    # is this the first one we find? then remove all previous "done" hypotheses,
                    # as they had failed queries
                    if not previously_found_hypothesis_without_failed_queries:
                        # print("found a hypothesis without failed queries, discarding", len(hypotheses_done))
                        hypotheses_done = [ ]

                if core_hyp.no_failed_core_constraints():
                    previously_found_hypothesis_without_failed_queries = True


                # mark this hypothesis as done
                hypotheses_done.append(core_hyp)
                # print("HIER", len(core_hyp.unfillable), core_hyp.lweight)# , [(q, v[-5:]) for q, v in core_hyp.qvar_filler.items()])
                # input()

                ## for q, v in core_hyp.qvar_filler.items():
                ##     print("qvar", q, v[-5:])
                ## print([s[-5:] for s in core_hyp.hypothesis.stmts])
                ## print("----")
                        
                continue

            new_hypotheses = core_hyp.extend()
            # put extensions of this hypothesis to the beginning of the queue, such that
            # we explore one hypothesis to the end before we start the next.
            # this way we can see early if we have hypotheses without failed queries
            hypotheses_todo.extendleft(new_hypotheses)
            # hypotheses_todo.extend(new_hypotheses)

        if not previously_found_hypothesis_without_failed_queries:
            print("Warning: All hypotheses had at least one failed query.")
        
        # at this point, all hypotheses are as big as they can be.
        return hypotheses_done

    def _make_qvar_signature(self, h):
        ##
        def make_one_signature(keys, qfdict):
            return "_".join(k + "|" + qfdict[k][-5:] for k in sorted(keys))
        ##
        
        if len(h.qvar_filler) - len(h.entrypoints) < self.QS_COUNT_CUTOFF:
            return None

        # make string characterizing entry points
        qs_entry = make_one_signature(h.entrypoints, h.qvar_filler)
        # and concatenate with string characterizing other fillers
        return [qs_entry + "_" + make_one_signature(keys, h.qvar_filler) for keys in itertools.combinations(sorted(k for k in h.qvar_filler.keys() if k not in h.entrypoints), 2)]
    
    #########################
    #########################
    # ENTRY POINT HANDLING
    
    #################################
    # Return any combination of entry point fillers for all the entry points
    #
    # returns pairs (qvar_filler, weight)
    # where qvar_filler is a dictionary mapping query variables to fillers, and
    # weight is the confidence of the fillers
    def _each_entry_point_combination(self, entrypoints, entrypoint_weights, facet):
        # variables occurring in this facet: query constraints have the form [subj, pred, obj] where subj, obj are variables.
        # collect those
        facet_variables = set(c[0] for c in facet["queryConstraints"]).union(c[2] for c in facet["queryConstraints"])
        # variables we are filling: all entry points that appear in the query constraints of this facet
        entrypoint_variables = sorted(e for e in entrypoints.keys() if e in facet_variables)

        # itertools.product does Cartesian product of n sets
        # here we do a product of entry point filler indices, so we can access each filler as well as its weight
        filler_index_tuples = [ ]
        weights = [ ]

        for filler_indices in itertools.product(*(range(len(entrypoints[v])) for v in entrypoint_variables)):

            # qvar-> filler mapping: pair each entry point variable with the i-th filler, where i
            # is the filler index for that entry point variable
            qvar_fillers = dict((v, entrypoints[v][i]) for v, i in zip(entrypoint_variables, filler_indices))

            # reject if any two variables are mapped to the same ERE
            if any (qvar_fillers[v1] == qvar_fillers[v2] for v1 in entrypoint_variables for v2 in entrypoint_variables if v1 != v2):
                continue
            
            filler_index_tuples.append( qvar_fillers)
            # weight:
            # filler weights are in the range of [0, 100]
            # multiply weights/100 of the fillers,
            # then take the log to be in log-probability space
            weight = 1
            for v, i in zip(entrypoint_variables, filler_indices):
                weight *= entrypoint_weights[v][i]
            weights.append(weight)
            # print("HIER", qvar_fillers, weight)
            #input()
            # weights.append( math.log(functools.reduce(operator.mul, (entrypoint_weights[v][i]/100.0 for v, i in zip(entrypoint_variables, filler_indices)))))

        for qvar_filler, weight in sorted(zip(filler_index_tuples, weights), key = lambda pair:pair[1], reverse = True):
            # print("HIER2wt", weight)
            yield (qvar_filler, weight)

    def _each_entry_point_combination_w_early_cutoff(self, entrypoints, entrypoint_weights, facet, earlycutoff):
        # variables occurring in this facet: query constraints have the form [subj, pred, obj] where subj, obj are variables.
        # collect those
        facet_variables = set(c[0] for c in facet["queryConstraints"]).union(c[2] for c in facet["queryConstraints"])
        # variables we are filling: all entry points that appear in the query constraints of this facet
        entrypoint_variables = sorted(e for e in entrypoints.keys() if e in facet_variables)

        ep_weight_filler_mapping = {}
        for ep_var in entrypoint_variables:
            ep_weight_filler_mapping[ep_var] = defaultdict(list)
            ep_fillers = entrypoints[ep_var]
            ep_weights = entrypoint_weights[ep_var]
            # Make sure all weights are positive (in case of using role weighting in SoIN matching,
            # there might be negative weights for entry points, which might mess up rankings)
            if min(ep_weights) < 0.1:
                ep_weights = [w - min(ep_weights) + 0.1 for w in ep_weights]
            for ep_filler, ep_weight in zip(ep_fillers, ep_weights):
                ep_weight_filler_mapping[ep_var][ep_weight].append(ep_filler)

        def ep_weights_to_ep_fillers(ep_weights):
            ep_vars = []
            fillers_list = []
            for ep_var, weight in ep_weights.items():
                ep_vars.append(ep_var)
                fillers_list.append(ep_weight_filler_mapping[ep_var][weight])
            ep_fillers_list = []
            for fillers in itertools.product(*fillers_list):
                # Filter cases where there are duplicate node ids for different entry points
                if len(set(fillers)) == len(fillers):
                    ep_fillers_list.append(
                        {ep_var: filler for ep_var, filler in zip(ep_vars, fillers)})
            return ep_fillers_list

        fillers_list = []
        weight_list = []

        # Group 1 (all highest)
        group_1_ep_weights = {}
        group_1_weight_prod = 1.0
        for ep_var, weight_filler_mapping in ep_weight_filler_mapping.items():
            var_weight = sorted(weight_filler_mapping.keys(), reverse=True)[0]
            group_1_ep_weights[ep_var] = var_weight
            group_1_weight_prod *= var_weight

        group_1_ep_fillers = ep_weights_to_ep_fillers(group_1_ep_weights)

        print('Found {} filler combinations with weight {} (group #1)'.format(
            len(group_1_ep_fillers), group_1_weight_prod))

        fillers_list.extend(group_1_ep_fillers)
        weight_list.extend([group_1_weight_prod] * len(group_1_ep_fillers))

        # Group 2 (all-but-one highest & one second-highest)
        if len(fillers_list) < earlycutoff:
            group_2_ep_weights_list = []
            group_2_weight_prod_list = []

            # For each entry point, select its second-highest weighted fillers
            for idx in range(len(ep_weight_filler_mapping)):
                ep_weights = {}
                weight_prod = 1.0
                for ep_var_idx, (ep_var, weight_filler_mapping) in enumerate(
                        ep_weight_filler_mapping.items()):
                    if ep_var_idx == idx:
                        if len(weight_filler_mapping) < 2:
                            continue
                        var_weight = sorted(weight_filler_mapping.keys(), reverse=True)[1]
                    else:
                        var_weight = sorted(weight_filler_mapping.keys(), reverse=True)[0]

                    ep_weights[ep_var] = var_weight
                    weight_prod *= var_weight

                if len(ep_weights) == len(ep_weight_filler_mapping):
                    group_2_ep_weights_list.append(ep_weights)
                    group_2_weight_prod_list.append(weight_prod)

            for ep_weights, weight_prod in sorted(
                    zip(group_2_ep_weights_list, group_2_weight_prod_list),
                    key=itemgetter(1), reverse=True):
                ep_fillers = ep_weights_to_ep_fillers(ep_weights)

                print('Found {} filler combinations with weight {} (group #2)'.format(
                    len(ep_fillers), weight_prod))

                fillers_list.extend(ep_fillers)
                weight_list.extend([weight_prod] * len(ep_fillers))

                if len(fillers_list) >= earlycutoff:
                    break

        # Group 3 (all-but-one highest & one third-highest,
        # or all-but-two highest & two second-highest)
        if len(fillers_list) < earlycutoff:
            group_3_ep_weights_list = []
            group_3_weight_prod_list = []

            # For each entry point, select its third-highest weighted fillers
            for idx in range(len(ep_weight_filler_mapping)):
                ep_weights = {}
                weight_prod = 1.0
                for ep_var_idx, (ep_var, weight_filler_mapping) in enumerate(
                        ep_weight_filler_mapping.items()):
                    if ep_var_idx == idx:
                        if len(weight_filler_mapping) < 3:
                            continue
                        var_weight = sorted(weight_filler_mapping.keys(), reverse=True)[2]
                    else:
                        var_weight = sorted(weight_filler_mapping.keys(), reverse=True)[0]

                    ep_weights[ep_var] = var_weight
                    weight_prod *= var_weight

                if len(ep_weights) == len(ep_weight_filler_mapping):
                    group_3_ep_weights_list.append(ep_weights)
                    group_3_weight_prod_list.append(weight_prod)

            # For each combination of 2 entry points, select their second-highest weighted fillers
            for idx_1, idx_2 in itertools.combinations(range(len(ep_weight_filler_mapping)), 2):
                ep_weights = {}
                weight_prod = 1.0
                for ep_var_idx, (ep_var, weight_filler_mapping) in enumerate(
                        ep_weight_filler_mapping.items()):
                    if ep_var_idx == idx_1 or ep_var_idx == idx_2:
                        if len(weight_filler_mapping) < 2:
                            continue
                        var_weight = sorted(weight_filler_mapping.keys(), reverse=True)[1]
                    else:
                        var_weight = sorted(weight_filler_mapping.keys(), reverse=True)[0]

                    ep_weights[ep_var] = var_weight
                    weight_prod *= var_weight

                if len(ep_weights) == len(ep_weight_filler_mapping):
                    group_3_ep_weights_list.append(ep_weights)
                    group_3_weight_prod_list.append(weight_prod)

            for ep_weights, weight_prod in sorted(
                    zip(group_3_ep_weights_list, group_3_weight_prod_list),
                    key=itemgetter(1), reverse=True):
                ep_fillers = ep_weights_to_ep_fillers(ep_weights)

                print('Found {} filler combinations with weight {} (group #3)'.format(
                    len(ep_fillers), weight_prod))

                fillers_list.extend(ep_fillers)
                weight_list.extend([weight_prod] * len(ep_fillers))

                if len(fillers_list) >= earlycutoff:
                    break

        # Group 4 (all-but-one highest & one forth-highest,
        # or all-but-two highest & one second-highest & one third-highest,
        # or all-but-three highest & three second-highest)
        if len(fillers_list) < earlycutoff:
            group_4_ep_weights_list = []
            group_4_weight_prod_list = []

            # For each entry point, select its forth-highest weighted fillers
            for idx in range(len(ep_weight_filler_mapping)):
                ep_weights = {}
                weight_prod = 1.0
                for ep_var_idx, (ep_var, weight_filler_mapping) in enumerate(
                        ep_weight_filler_mapping.items()):
                    if ep_var_idx == idx:
                        if len(weight_filler_mapping) < 4:
                            continue
                        var_weight = sorted(weight_filler_mapping.keys(), reverse=True)[3]
                    else:
                        var_weight = sorted(weight_filler_mapping.keys(), reverse=True)[0]

                    ep_weights[ep_var] = var_weight
                    weight_prod *= var_weight

                if len(ep_weights) == len(ep_weight_filler_mapping):
                    group_4_ep_weights_list.append(ep_weights)
                    group_4_weight_prod_list.append(weight_prod)

            # For each permutation of 2 entry points, select the third-highest weighted fillers for one of them
            # and the second-highest weighted fillers for the other
            for idx_1, idx_2 in itertools.permutations(range(len(ep_weight_filler_mapping)), 2):
                ep_weights = {}
                weight_prod = 1.0
                for ep_var_idx, (ep_var, weight_filler_mapping) in enumerate(
                        ep_weight_filler_mapping.items()):
                    if ep_var_idx == idx_1:
                        if len(weight_filler_mapping) < 3:
                            continue
                        var_weight = sorted(weight_filler_mapping.keys(), reverse=True)[2]
                    elif ep_var_idx == idx_2:
                        if len(weight_filler_mapping) < 2:
                            continue
                        var_weight = sorted(weight_filler_mapping.keys(), reverse=True)[1]
                    else:
                        var_weight = sorted(weight_filler_mapping.keys(), reverse=True)[0]

                    ep_weights[ep_var] = var_weight
                    weight_prod *= var_weight

                if len(ep_weights) == len(ep_weight_filler_mapping):
                    group_4_ep_weights_list.append(ep_weights)
                    group_4_weight_prod_list.append(weight_prod)

            # For each combination of 3 entry points, select their second-highest weighted fillers
            for idx_1, idx_2, idx_3 in itertools.combinations(range(len(ep_weight_filler_mapping)), 3):
                ep_weights = {}
                weight_prod = 1.0
                for ep_var_idx, (ep_var, weight_filler_mapping) in enumerate(
                        ep_weight_filler_mapping.items()):
                    if ep_var_idx == idx_1 or ep_var_idx == idx_2 or ep_var_idx == idx_3:
                        if len(weight_filler_mapping) < 2:
                            continue
                        var_weight = sorted(weight_filler_mapping.keys(), reverse=True)[1]
                    else:
                        var_weight = sorted(weight_filler_mapping.keys(), reverse=True)[0]

                    ep_weights[ep_var] = var_weight
                    weight_prod *= var_weight

                if len(ep_weights) == len(ep_weight_filler_mapping):
                    group_4_ep_weights_list.append(ep_weights)
                    group_4_weight_prod_list.append(weight_prod)

            for ep_weights, weight_prod in sorted(
                    zip(group_4_ep_weights_list, group_4_weight_prod_list),
                    key=itemgetter(1), reverse=True):
                ep_fillers = ep_weights_to_ep_fillers(ep_weights)

                print('Found {} filler combinations with weight {} (group #3)'.format(
                    len(ep_fillers), weight_prod))

                fillers_list.extend(ep_fillers)
                weight_list.extend([weight_prod] * len(ep_fillers))

                if len(fillers_list) >= earlycutoff:
                    break

        for fillers, weight in zip(fillers_list, weight_list):
            yield(fillers, weight)

    def _entrypoint_filler_rolescore(self, ep_var, ep_filler, facet):
        score = 0
        
        for subj, pred, obj in facet["queryConstraints"]:
            if subj == ep_var:
                # look for roles adjacent to ep_filler with predicate pred and ep_filler as the subject
                if any(stmt for stmt in self.graph_obj.each_ere_adjacent_stmt(ep_filler, pred, "subject")):
                    score += 1
            elif obj == ep_var:
                # look for roles adjacent to ep_filler with predicate pred and ep_filler as the object
                if any(stmt for stmt in self.graph_obj.each_ere_adjacent_stmt(ep_filler, pred, "object")):
                    score += 1


        return score
    
    #########################
    #########################
    # TEMPORAL ANALYSIS
    ##################################
    # given the "temporal" piece of a statement of information need,
    # turn the date and time info in the dictionary
    # into Python datetime objects
    def _pythonize_datetime(self, json_temporal):
        retv = { }
        for qvar, tconstraint in json_temporal.items():
            retv[qvar] = { }
            
            if "start_time" in tconstraint:
                entry = tconstraint["start_time"]
                retv[qvar]["start_time"] = AidaIncompleteDate(entry.get("year", None), entry.get("month", None), entry.get("day", None))
            if "end_time" in tconstraint:
                entry = tconstraint["end_time"]
                retv[qvar]["end_time"] = AidaIncompleteDate(entry.get("year", None), entry.get("month", None), entry.get("day", None))
                                                        
        return retv

    #########################
    #########################
    # RANKING
    #########################
    # compute a weight for all cluster seeds in self.hypotheses
    def _rank_seeds(self):
        return self._sort_hypotheses_grouped(self.hypotheses, lambda h: h.entrypointweight, self._sort_hypotheses_lweight_connectedness_novelty, "level 1")
    
    # grouping hypotheses by weight,
    # then sorting each group,
    # then concatenate results.
    # high weight first
    def _sort_hypotheses_grouped(self, hypotheses, weight_function, sorting_function, label):
        groups = { }
        for h in hypotheses:
            weight = weight_function(h)
            if weight not in groups:
                groups[weight] = [ ]
            groups[weight].append(h)

        sorted_hypotheses =  [ ]
        for weight, group in sorted(groups.items(), key = lambda pair: pair[0], reverse = True):
            # print("group" ,label, "with weight", weight, len(group))# , [(v, h.qvar_filler[v][-5:]) for v in h.entrypoints for h in group])
            sorted_hypotheses += sorting_function(group)

        return sorted_hypotheses

    def _sort_hypotheses_lweight_connectedness_novelty(self, hypotheses):
        # group hypotheses by lweight, then sort groups by connectedness
        return self._sort_hypotheses_grouped(hypotheses, lambda h:h.lweight, self._sort_hypotheses_connectedness_novelty, "level 2")


    def _sort_hypotheses_connectedness_novelty(self, hypotheses):
        sorted_hypotheses = self._sort_hypotheses_connectedness(hypotheses)
        return self._rerank_hypotheses_novelty(sorted_hypotheses)
    
    def _sort_hypotheses_connectedness(self, hypotheses):
        weights = [ ]

        for hypothesis in hypotheses:
            outdeg = 0
            # for each ERE of this hypothesis
            for erelabel in hypothesis.hypothesis.eres():
                # find statements IDs of statements adjacent to the EREs of this hypothesis
                for stmtlabel in self.graph_obj.each_ere_adjacent_stmt_anyrel(erelabel):
                    outdeg += 1
            weights.append( outdeg)

        # print("connectedness weights", weights)
        return [pair[1] for pair in sorted(enumerate(hypotheses), key = lambda pair:weights[pair[0]], reverse = True)]


    def _rerank_hypotheses_novelty(self, hypotheses):
        if len(hypotheses) == 0:
            return hypotheses
        
        done = [ hypotheses[0] ]
        todo = hypotheses[1:]

        qvar_characterization = self._update_qvar_characterization_for_seednovelty({ }, hypotheses[0])

        
        while len(todo) > max(0, len(hypotheses) - self.rank_first_k):
            # select the item to rank next
            # print("HIER qvar ch.", qvar_characterization)
            nextitem_index = self._choose_seed_novelty_one(todo, qvar_characterization)
            if nextitem_index is None:
                # we didn't find any more items to rank
                break
            
            # append the next best item to the ranked items
            nextitem = todo.pop(nextitem_index)
            # print("next is", nextitem_index)
            done.append(nextitem)
            qvar_characterization = self._update_qvar_characterization_for_seednovelty(qvar_characterization, nextitem)

        # at this point we have ranked the self.rank_first_k items
        # just attach the rest of the items at the end
        done += todo
        # print("DONE")

        return done
    
    def _choose_seed_novelty_one(self, torank, qvar_characterization):
        # for each item in torank, determine difference from items in ranked
        # in terms of qvar_filler

        best_index = None
        best_value = None
        
        for index, hyp in enumerate(torank):
            if index >= self.consider_next_k_in_reranking:
                # we have run out of the next k to consider,
                # don't go further down the list
                break

            this_value = 0
            for qvar, filler in hyp.qvar_filler.items():
                if qvar in hyp.entrypoints:
                    # do not count entry point variables when checking for novelty
                    # print("skipping variable in ranking", qvar)
                    continue
                if qvar in qvar_characterization.keys():
                    if filler in qvar_characterization[ qvar ]:
                        # there are higher-ranked hypotheses that have the same filler
                        # for this qvar. take a penalty for that
                        this_value += qvar_characterization[qvar][filler]
                    else:
                        # novel qvar filler! Take a bonus
                        this_value += self.bonus_for_novelty
                else:
                    # this hypothesis, for some reason, has a query variable
                    # that we haven't seen before.
                    # this shouldn't happen.
                    # oh well, take a bonus for novelty then
                    this_value += self.bonus_for_novelty

            # print("HIER1", this_value, hyp.qvar_filler)
            # input("hit enter...")

            # at this point we have the value for the current hypothesis.
            # if it is the minimum achievable value, stop here and go with this hypothesis
            if this_value <= self.bonus_for_novelty * len(qvar_characterization):
                best_index = index
                best_value = this_value
                break

            # check if the current value is better than the previous best.
            # if so, record this index as the best one
            if best_value is None or this_value < best_value:
                best_index = index
                best_value = this_value


        return best_index

        
    # ranking by seed novelty uses a characterization of the query variable fillers for the
    # already ranking items.
    # this function takes an existing query variable characterization and updates it
    # with the query variable fillers of the given hypothesis, which is a OneClusterSeed.
    # format of qvar_characterization:
    # qvar -> filler -> count
    #
    # That is, we penalize a hypothesis that has the same qvar filler that we have seen before
    # with a value equivalent to the number of previous hypotheses that had the same filler.
    def _update_qvar_characterization_for_seednovelty(self, qvar_characterization, hypothesis):
        for qvar, filler in hypothesis.qvar_filler.items():
            if qvar in hypothesis.entrypoints:
                # do not include entry points in the novelty calculation:
                # novelty in entry points is not rewarded.
                continue
            
            if qvar not in qvar_characterization:
                qvar_characterization[ qvar ] = { }

            qvar_characterization[ qvar ][ filler ] = qvar_characterization[qvar].get(filler, 0) + 1

        return qvar_characterization

    ## # ranking by weight and connectedness:
    ## # group hypotheses by weight, then rank equal-weight hypotheses by connectedness
    ## def _rankby_weight_and_connectedness(self, hypotheses):
    ##     hgroups = self._group_seeds_byweight(hypotheses)

    ##     # list of ranked hypotheses, highest first
    ##     ranked_hypotheses =  [ ]
    ##     for weight, hgroup in sorted(hgroups.items(), key = lambda pair: pair[1], reverse = True):
    ##         ranking_levels = self._rank_seed_connectedness(hgroup)
    ##         hypotheses_sorted = [pair[1] for pair in sorted(enumerate(self.hypotheses), key = lambda pair:ranking_levels[pair[0]], reverse = True)]
    ##         ranked_hypotheses += hypotheses_sorted

    ##     return ranked_hypotheses
        

    ## # rank hypotheses by their lweight,
    ## # which indicates how good their entrypoints were,
    ## # and whether they failed to meet any query constraints
    ## #
    ## # returns ranks of hypotheses, highest should go first
    ## def _rank_seed_byweight(self, hypotheses):
    ##     # sort hypotheses by weight, highest first
    ##     # return ranks
    ##     # highest to lowest, hence listlength-ranking
    ##     return (len(hypotheses) - scipy.stats.rankdata([h.lweight for h in hypotheses], method = "min")).astype(int)
    
    
    ## # weighting based on connectedness of a cluster seed:
    ## # sum of degrees of EREs in the cluster.
    ## # this rewards both within-cluster and around-cluster connectedness
    ## # this does seed connectedness ratings for multiple groups of hypotheses,
    ## # where the grouping is a mapping from weight to group
    ## def _rank_seed_connectedness(self, hypotheses):
    ##     weights = [ ]

    ##     for hypothesis in hypotheses:
    ##         outdeg = 0
    ##         # for each ERE of this hypothesis
    ##         for erelabel in hypothesis.hypothesis.eres():
    ##             # find statements IDs of statements adjacent to the EREs of this hypothesis
    ##             for stmtlabel in self.graph_obj.each_ere_adjacent_stmt_anyrel(erelabel):
    ##                 outdeg += 1
    ##         weights.append( outdeg)

    ##     # print("connectedness weights", weights)

    ##     return (len(hypotheses) - scipy.stats.rankdata(weights, method = "min")).astype(int)

    ## # given a list of pairs (ranking, OneClusterSeed object),
    ## # produce a new such list where objects are ranked more highly
    ## # if they differ most from all the top k items
    ## def _rank_seed_novelty(self, hypotheses):
    ##     if len(hypotheses) == 0:
    ##         return hypotheses
        
    ##     ranked = [ hypotheses[0] ]
    ##     torank = hypotheses[1:]

    ##     qvar_characterization = self._update_qvar_characterization_for_seednovelty({ }, hypotheses[0])

    ##     # print("HIER rank0", hypotheses[0].qvar_filler)
        
    ##     while len(torank) > max(0, len(hypotheses) - self.rank_first_k):
    ##         # select the item to rank next
    ##         # print("HIER qvar ch.", qvar_characterization)
    ##         nextitem_index = self._rank_seed_novelty_one(torank, qvar_characterization)
    ##         if nextitem_index is None:
    ##             # we didn't find any more items to rank
    ##             break
            
    ##         # append the next best item to the ranked items
    ##         nextitem = torank.pop(nextitem_index)
    ##         ranked.append(nextitem)
    ##         qvar_characterization = self._update_qvar_characterization_for_seednovelty(qvar_characterization, nextitem)

    ##     # at this point we have ranked the self.rank_first_k items
    ##     # just attach the rest of the items at the end
    ##     ranked += torank

    ##     return ranked

    ## def _rank_seed_novelty_one(self, torank, qvar_characterization):
    ##     # for each item in torank, determine difference from items in ranked
    ##     # in terms of qvar_filler

    ##     best_index = None
    ##     best_value = None
        
    ##     for index, hyp in enumerate(torank):
    ##         if index >= self.consider_next_k_in_reranking:
    ##             # we have run out of the next k to consider,
    ##             # don't go further down the list
    ##             break

    ##         this_value = 0
    ##         for qvar, filler in hyp.qvar_filler.items():
    ##             if qvar in hyp.entrypoints:
    ##                 # do not count entry point variables when checking for novelty
    ##                 # print("skipping variable in ranking", qvar)
    ##                 continue
    ##             if qvar in qvar_characterization.keys():
    ##                 if filler in qvar_characterization[ qvar ]:
    ##                     # there are higher-ranked hypotheses that have the same filler
    ##                     # for this qvar. take a penalty for that
    ##                     this_value += qvar_characterization[qvar][filler]
    ##                 else:
    ##                     # novel qvar filler! Take a bonus
    ##                     this_value += self.bonus_for_novelty
    ##             else:
    ##                 # this hypothesis, for some reason, has a query variable
    ##                 # that we haven't seen before.
    ##                 # this shouldn't happen.
    ##                 # oh well, take a bonus for novelty then
    ##                 this_value += self.bonus_for_novelty

    ##         # print("HIER1", this_value, hyp.qvar_filler)
    ##         # input("hit enter...")

    ##         # at this point we have the value for the current hypothesis.
    ##         # if it is the minimum achievable value, stop here and go with this hypothesis
    ##         if this_value <= self.bonus_for_novelty * len(qvar_characterization):
    ##             best_index = index
    ##             best_value = this_value
    ##             break

    ##         # check if the current value is better than the previous best.
    ##         # if so, record this index as the best one
    ##         if best_value is None or this_value < best_value:
    ##             best_index = index
    ##             best_value = this_value


    ##     return best_index

        
    ## # ranking by seed novelty uses a characterization of the query variable fillers for the
    ## # already ranked items.
    ## # this function takes an existing query variable characterization and updates it
    ## # with the query variable fillers of the given hypothesis, which is a OneClusterSeed.
    ## # format of qvar_characterization:
    ## # qvar -> filler -> count
    ## #
    ## # That is, we penalize a hypothesis that has the same qvar filler that we have seen before
    ## # with a value equivalent to the number of previous hypotheses that had the same filler.
    ## def _update_qvar_characterization_for_seednovelty(self, qvar_characterization, hypothesis):
    ##     for qvar, filler in hypothesis.qvar_filler.items():
    ##         if qvar in hyothesis.entrypoints:
    ##             # do not include entry points in the novelty calculation:
    ##             # novelty in entry points is not rewarded.
    ##             continue
            
    ##         if qvar not in qvar_characterization:
    ##             qvar_characterization[ qvar ] = { }

    ##         qvar_characterization[ qvar ][ filler ] = qvar_characterization[qvar].get(filler, 0) + 1

    ##     return qvar_characterization
