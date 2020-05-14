# Katrin Erk April 2019
#
# Class for filtering hypotheses for logical consistency
# Rule-based filtering

from collections import deque

from aida_utexas.seeds.aidahypothesis import AidaHypothesis


class AidaHypothesisFilter:
    def __init__(self, thegraph):
        self.graph_obj = thegraph


    ##############################################################
    # Tests
    # They take in a hypothesis-under-construction, the statement (which is part of the hypothesis) to test,
    # and the full hypothesis (if that is available).
    #
    # Some filters only work post-hoc, namely the ones that make use of the full hypothesis.
    #
    # They return False if there is a problem, and True otherwise
    # Assumed invariant: the hypothesis is error-free except for possibly this statement.
    # (This can be achieved by filtering as each new statement is added)
    
    #######
    #
    ## # All attackers in a conflict.attack event need to have one possible affiliation in common,
    ## # also all instruments,
    ## # and all attackers and instruments
    ## # (if there is no known affiliation that is also okay).
    ## # Entities that are possible affiliates of any affiliation relation
    ## # are counted as their own affiliates.
    ## # For example, Ukraine counts as being affiliated with Ukraine.
    #
    # This filter does not use the full hypothesis, and hence can be used during hypothesis construction
    def event_attack_attacker_instrument_compatible(self, hypothesis, test_stmt, fullhypothesis = None):

        # is stmt an event role of a conflict.attack event, specifically an attacker or instrument?
        if not self.graph_obj.is_eventrole_stmt(test_stmt):
            return True
        if not self.graph_obj.stmt_predicate(test_stmt).startswith("Conflict.Attack"):
            return True
        if not self.graph_obj.stmt_predicate(test_stmt).endswith("Instrument") or self.graph_obj.stmt_predicate(test_stmt).endswith("Attacker"):
            return True

        # this is an event role of a conflict.attack.
        # its subject is the event ERE.
        event_ere = self.graph_obj.stmt_subject(test_stmt)
        
        # getting argument EREs for attackers and instruments
        attackers = list(hypothesis.eventrelation_each_argument_labeled_like(event_ere, "Conflict.Attack", "Attacker"))
        instruments = list(hypothesis.eventrelation_each_argument_labeled_like(event_ere, "Conflict.Attack", "Instrument"))

        # if there are multiple attackers but no joint affiliation: problem
        attacker_affiliations_intersect = self._possible_affiliation_intersect(hypothesis, attackers)
        if attacker_affiliations_intersect is not None and len(attacker_affiliations_intersect) == 0:
            # print("HIER1 no intersection between attacker affiliations")
            return False

        instrument_affiliations_intersect = self._possible_affiliation_intersect(hypothesis, instruments)
        if instrument_affiliations_intersect is not None and len(instrument_affiliations_intersect) == 0:
            # print("HIER2 no intersection between instrument affiliations")
            return False
        
        if attacker_affiliations_intersect is not None and instrument_affiliations_intersect is not None and len(attacker_affiliations_intersect.intersection(instrument_affiliations_intersect)) == 0:
            # print("no intersection betwen attacker and instrument affiliations", attacker_affiliations_intersect, instrument_affiliations_intersect)
            return False

        # no problem here
        return True

    #######
    #
    ## All roles of an attack event need to be filled by different fillers
    # (so, no attacking yourself)
    #
    # This filter does not use the full hypothesis, and hence can be used during hypothesis construction
    def event_attack_all_roles_different(self, hypothesis, test_stmt, fullhypothesis = None):
        # is stmt an event role of a conflict.attack event, specifically an attacker or instrument?
        if not self.graph_obj.is_eventrole_stmt(test_stmt):
            return True
        if not self.graph_obj.stmt_predicate(test_stmt).startswith("Conflict.Attack"):
            return True

        event_ere = self.graph_obj.stmt_subject(test_stmt)
        arg_ere = self.graph_obj.stmt_object(test_stmt)

        # print("HIER2", [(stmt, label, stmtarg, stmtarg == arg_ere, stmt == test_stmt) for stmt, label, stmtarg in hypothesis.eventrelation_each_argstmt(event_ere)])
        if any(stmtarg == arg_ere for stmt, label, stmtarg in hypothesis.eventrelation_each_argstmt(event_ere) if stmt != test_stmt):
            # print("two attack roles filled by same argument, discarding", test_stmt)
            return False

        return True
        
    #######
    #
    # Don't have multiple types on an event or relation.
    #
    # This filter takes the full hypothesis into account and hence only works post-hoc.
    def single_type_per_eventrel(self, hypothesis, test_stmt, fullhypothesis):
        # potential problem only if this is a type statement
        if not self.graph_obj.is_typestmt(test_stmt):
            return True

        ere = self.graph_obj.stmt_subject(test_stmt)

        # no problem if we have an entity
        if self.graph_obj.is_entity(ere):
            return True

        # okay, we have an event or relation.
        # check whether this ere has another type
        types = [typelabel for typelabel in hypothesis.ere_each_type(ere)]
        if len(types) > 1:
            # print("more than one type statement, flagging", test_stmt, self.graph_obj.stmt_object(test_stmt), types)
            return False

        # this is an only type. Check to see whether it coincides with any roles of this event or relation
        # IN THE FULL HYPOTHESIS (this is the part that only works post-hoc)
        eventrel_roles_ere = [ self.graph_obj.shorten_label(rolelabel) for rolelabel, arg in fullhypothesis.eventrelation_each_argument(ere) ]
        typelabel = self.graph_obj.shorten_label(self.graph_obj.stmt_object(test_stmt))
        if not(any(rolelabel.startswith(typelabel) and len(rolelabel) > len(typelabel) for rolelabel in eventrel_roles_ere)):
            # no, this type label does not coincide with any role label
            # print("type statement not matching any role, flagging", test_stmt, self.graph_obj.stmt_object(test_stmt), eventrel_roles_ere)
            return False

        return True
        
    #######
    #
    # Don't have relations with only one argument.
    #
    # This filter takes the full hypothesis into account and hence only works post-hoc.
    def relations_need_twoargs(self, hypothesis, test_stmt, fullhypothesis):
        # is this an argument of a relation
        if not(self.graph_obj.is_relation(self.graph_obj.stmt_subject(test_stmt)) and self.graph_obj.is_ere(self.graph_obj.stmt_object(test_stmt))):
            # no: then we don't have a problem
            return True

        rel_ere = self.graph_obj.stmt_subject(test_stmt)

        # check if this relation ERE has more than one argument IN THE FULL HYPOTHESIS (this is the part
        # that only works post-hoc)
        rel_roles = set(rolelabel for rolelabel, arg in fullhypothesis.eventrelation_each_argument(rel_ere))
        if len(rel_roles) > 1:
            # print("keeping arg of role", rel_ere, rel_roles)
            return True

        # this is the only argument of this relation. don't add it
        return False
    
    #######
    #
    # Don't have events with only one argument, except when they are core EREs (that is, adjacent to
    # one of the core statements)
    #
    # This filter takes the full hypothesis into account and hence only works post-hoc.
    def events_need_twoargs(self, hypothesis, test_stmt, fullhypothesis):
        # is this an argument of a relation
        if not(self.graph_obj.is_event(self.graph_obj.stmt_subject(test_stmt)) and self.graph_obj.is_ere(self.graph_obj.stmt_object(test_stmt))):
            # no: then we don't have a problem
            return True

        event_ere = self.graph_obj.stmt_subject(test_stmt)
        arg_ere = self.graph_obj.stmt_object(test_stmt)

        if arg_ere in hypothesis.core_eres():
            # then it's fine, keep the one-argument event
            # print("keeping one-argument event, as the argument is a core ERE", test_stmt)
            return True

        # check if this event ERE has more than one argument IN THE FULL HYPOTHESIS (this is the part
        # that only works post-hoc)
        event_roles = set(rolelabel for rolelabel, arg in fullhypothesis.eventrelation_each_argument(event_ere))
        if len(event_roles) > 1:
            return True

        # this is the only argument of this relation. don't add it
        return False
    
    ##########################################
    # main checking function
    # check one single statement, which is part of the hypothesis.
    # assumption: this statement is the only potentially broken statement in the hypothesis
    def validate(self, hypothesis, stmt, fullhypothesis = None):

        if fullhypothesis is None:
            # interactive filtering, only use tests usable for that
            # print("interactive filtering")
            tests = [
                self.event_attack_attacker_instrument_compatible,
                self.event_attack_all_roles_different
                ]

            for test_okay in tests:
                if not test_okay(hypothesis, stmt):
                    return False

        else:
            # print("posthoc filtering")
            tests = [
                self.event_attack_attacker_instrument_compatible,
                self.event_attack_all_roles_different,
                self.single_type_per_eventrel,
                self.relations_need_twoargs,
                self.events_need_twoargs
                ]

            for test_okay in tests:
                if not test_okay(hypothesis, stmt, fullhypothesis):
                    return False
                
        return True

    #############################################
    # other main function:
    # post-hoc, remove statements from the hypothesis that shouldn't be there.
    # do this by starting a new hypothesis and re-inserting statements there by statement weight,
    # using the validate function
    def filtered(self, hypothesis):
        
        # new hypothesis: "incremental" because we add in things one at a time.
        # start with the core statements
        incr_hypothesis = AidaHypothesis(self.graph_obj, stmts = hypothesis.core_stmts.copy(),
                                             core_stmts = hypothesis.core_stmts.copy(),
                                             stmt_weights = dict((stmt, wt) for stmt, wt in hypothesis.stmt_weights.items() if stmt in hypothesis.core_stmts),
                                             lweight = hypothesis.lweight)
        incr_hypothesis.add_failed_queries(hypothesis.failed_queries)
        incr_hypothesis.add_qvar_filler(hypothesis.qvar_filler)
        incr_hypothesis_eres = set(incr_hypothesis.eres())

        # print("HIER0", sorted(incr_hypothesis.stmts))

        # all other statements are candidates, sorted by their weights in the hypothesis, highest first
        candidates = [ stmt for stmt in hypothesis.stmts if stmt not in hypothesis.core_stmts]
        candidates.sort(key = lambda stmt:hypothesis.stmt_weights[stmt], reverse = True)
        candidates = deque(candidates)

        # candidates are set aside if they currently don't connect to any ERE in the incremental hypothesis
        candidates_set_aside = deque()

        while len(candidates) > 0:
            ##
            # any set-aside candidates that turn out to be connected to the hypothesis after all?
            resurrected_stmts = [ ]
            for stmt in candidates_set_aside:
                if any(ere in incr_hypothesis_eres for ere in self.graph_obj.statement_args(stmt)):
                    # yes, check now whether this candidate should be inserted
                    # print("resurrecting", stmt)
                    resurrected_stmts.append(stmt)
                    incr_hypothesis, new_eres = self._test_and_insert_candidate(stmt, incr_hypothesis, hypothesis)
                    incr_hypothesis_eres.update(new_eres)
                    # print("HIER2", sorted(incr_hypothesis.stmts))
            for stmt in resurrected_stmts: candidates_set_aside.remove ( stmt )

            # now test the next non-set-aside candidate
            stmt = candidates.popleft()
            # does it need to be set aside?
            if not(self.graph_obj.stmt_subject(stmt) in incr_hypothesis_eres) and not(self.graph_obj.stmt_object(stmt) in incr_hypothesis_eres):
                # print("setting aside", stmt)
                candidates_set_aside.append(stmt)
            else:
                # no, we can test this one now.
                incr_hypothesis, new_eres = self._test_and_insert_candidate(stmt, incr_hypothesis, hypothesis)
                incr_hypothesis_eres.update(new_eres)

            # print("HIER", sorted(incr_hypothesis.stmts))

        # no candidates left in the candidate set, but maybe something from the set-aside candidate list
        # has become connected to the core by the last candidate to be added
        for stmt in candidates_set_aside:
            if any(ere in incr_hypothesis_eres for ere in self.graph_obj.statement_args(stmt)):
                # yes, check now whether this candidate should be inserted
                # print("resurrecting", stmt)
                incr_hypothesis, new_eres = self._test_and_insert_candidate(stmt, incr_hypothesis, hypothesis)
                incr_hypothesis_eres.update(new_eres)
                # print("HIER2", sorted(hypothesis.stmts))        

        return incr_hypothesis
        


    def _test_and_insert_candidate(self, stmt, prev_hypothesis, full_hypothesis):
        testhypothesis = prev_hypothesis.extend(stmt, weight = full_hypothesis.stmt_weights[stmt])

        new_eres = set()
        if self.validate(testhypothesis, stmt, full_hypothesis):
            # yes, statement is fine. add the statement's EREs to the incremental EREs
            # print("keeping stmt", stmt)
            for nodelabel in [ self.graph_obj.stmt_subject(stmt), self.graph_obj.stmt_object(stmt) ]:
                if self.graph_obj.is_ere(nodelabel):
                    new_eres.add(nodelabel)
                # retain the extended hypothesis
                return (testhypothesis, new_eres)
        else:
            # don't add stmt after all
            return (prev_hypothesis, new_eres)
        

    #######3
    # helper functions
    
    # intersection of possible affiliation IDs of EREs.
    # returns None if no known affiliations
    #
    # input: list of filler EREs
    def _possible_affiliation_intersect(self, hypothesis, ere_ids):
        affiliations = None
        
        for ere_id in ere_ids:
            these_affiliations = set(hypothesis.ere_each_possible_affiliation(ere_id))
            if hypothesis.ere_possibly_isaffiliation(ere_id):
                these_affiliations.add(ere_id)
            if len(these_affiliations) > 0:
                if affiliations is None:
                    affiliations = these_affiliations
                else:
                    affiliations.intersection_update(these_affiliations)

        return affiliations
