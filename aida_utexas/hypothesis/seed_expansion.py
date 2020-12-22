# Katrin Erk December 2020
# first rought stab at query expansion for hypothesis seeds

import json
import logging

class HypothesisSeedExpansion:
    def __init__(self):
        # hand-crafted paraphrases
        self.paraphrasing_json = {
            # simple paraphrase:
            # just replace one event type by another,
            # with a mapping on arguments
            # We only specify this from the point of view of Conflict.Attack,
            # assuming that paraphrasing is symmetric and transitive.
            # Expansion to all paraphrase pairs needs to happen in the code
            "SimpleParaphrase" : {
                "Conflict.Attack" : {
                    "Life.Die.DeathCausedByViolentEvents" : {
                        "Attacker": "Killer",
                        "Target": "Victim", 
                        "Instrument": "Instrument",
                        "Place": "Place"},
                    "Life.Injure": {
                        "Attacker": "Injurer",
                        "Target": "Victim",
                        "Instrument": "Instrument",
                        "Place": "Place" },
                    "Life.Injure.InjuryCausedByViolentEvents" : {
                        "Attacker": "Injurer",
                        "Target": "Victim",
                        "Instrument": "Instriment",
                        "Place": "Place"}, 
                    "Conflict.Attack.FirearmAttack" : {
                        "Attacker": "Attacker",
                        "Target": "Target",
                        "Instrument": "Instrument",
                        "Place": "Place"}, 
                    "Conflict.Attack.Bombing" : {
                        "Attacker": "Attacker",
                        "Target": "Target",
                        "Instrument": "Instrument",
                        "Place": "Place" }, 
                    "Conflict.Attack.AirstrikeMissileStrike" : {
                        "Attacker": "Attacker",
                        "Target": "Target",
                        "Instrument": "Instrument",
                        "Place": "Place"}, 
                    "Conflict.Attack.BiologicalChemicalPoisonAttack": {
                        "Attacker": "Attacker",
                        "Target": "Target",
                        "Instrument": "Instrument",
                        "Place": "Place"}, 
                    "Conflict.Attack.SelfDirectedBattle": {
                        "Attacker": "Attacker",
                        "Target": "Target",
                        "Instrument": "Instrument",
                        "Place": "Place" }, 
                    "Conflict.Attack.SetFire": {
                        "Attacker": "Attacker",
                        "Target": "Target",
                        "Instrument": "Instrument",
                        "Place": "Place"},
                    "Conflict.Attack.Stabbing": {
                        "Attacker": "Attacker",
                        "Target": "Target",
                        "Instrument": "Instrument",
                        "Place": "Place" }, 
                    "Conflict.Attack.Strangling": {
                        "Attacker": "Attacker",
                        "Target": "Target",
                        "Instrument": "Instrument",
                        "Place": "Place" }
                    }
                },
            # complex cause:
            # we have an event ?E, and a constraint that asks, say, for
            # Responsibility_Event ?E and
            # Responsibility_ResponsibleEntity ?X
            #
            # where ?E has type EType, transform to
            # ?E EType_Agent ?X
            # encode as:
            # Responsibility ->  "Event" -> eventrolelabel, "Entity" -> entityrolelabel
            # plus have, for events whose EventType may apply, a list of agent roles
            "ComplexCause": {
                "EventTypes": {
                    "ResponsibilityBlame.AssignBlame.AssignBlame": {
                        "Event" : "Event",
                        "Entity" : "EntityResponsible"},
                    "GeneralAffiliation.Sponsorship": {
                        "Event" : "ActorOrEvent",
                        "Entity" : "Sponsor"},
                    "GeneralAffiliation.Sponsorship.HelpSupport": {
                        "Event" : "ActorOrEvent",
                        "Entity" : "Sponsor"}},
                "AgentRole" : {
                        "Conflict.Attack":"Attacker",
                        "Life.Die.DeathCausedByViolentEvents" : "Killer",
                        "Life.Injure": "Injurer",
                        "Life.Injure.InjuryCausedByViolentEvents" : "Injurer",
                        "Conflict.Attack.FirearmAttack" : "Attacker",
                        "Conflict.Attack.Bombing" : "Attacker",
                        "Conflict.Attack.AirstrikeMissileStrike" : "Attacker",
                        "Conflict.Attack.BiologicalChemicalPoisonAttack": "Attacker",
                        "Conflict.Attack.SelfDirectedBattle": "Attacker",
                        "Conflict.Attack.SetFire": "Attacker",
                        "Conflict.Attack.Stabbing": "Attacker",
                        "Conflict.Attack.Strangling": "Attacker",
                        "Conflict.Demonstrate.MarchProtestPoliticalGathering" : "Demonstrator",
                        "Conflict.Coup.Coup" : "DeposingEntity"}}
        }


    # internal function: given an event type,
    # identify other event types that work as paraphrases,
    # with role mappings
    def _has_simple_paraphrases(self, eventtarget):
        # mapping other_eventtype -> { myrolelabel -> otherrolelabel}
        retv = { }
        # there is an entry in the simple paraphrasing list where this event type is the head
        if eventtarget in self.paraphrasing_json["SimpleParaphrase"]:
            retv.update(self.paraphrasing_json["SimpleParaphrase"][eventtarget])

        # if there is an entry in th simple paraphrasing list where this event type is the tail:
        # rephrase paraphrasings to make this event type the head
        for eventhead in self.paraphrasing_json["SimpleParaphrase"]:
            hpar = self.paraphrasing_json["SimpleParaphrase"][eventhead]
            if eventtarget in hpar:
                # make entry for event head, with reversed role labels
                retv[ eventhead] = dict( (myrole, headrole) for headrole, myrole in hpar[eventtarget].items())
                # make entries for all the paraphrases of the event head, except myself
                for eventtype in hpar.keys():
                    if eventtype == eventtarget:
                        continue
                    retv[eventtype] = dict( (myrole, hpar[eventtype][headrole]) for myrole, headrole in retv[eventhead].items())
        return retv

    # given an event type that could signal a complex cause,
    # check whether we really have a match.
    # if so, make a paraphrase
    def _complexcause(self, facetlabel, eventlabel, eventtype, coreconstraints):
        if eventtype not in self.paraphrasing_json["ComplexCause"]["EventTypes"].keys():
            return []

        # logging.info(f'found ccause event type {eventtype}')

        # what is the role label of the role whose argument is an event, if any?
        eventrole = eventtype + "_" + self.paraphrasing_json["ComplexCause"]["EventTypes"][eventtype]["Event"]

        # find the filler of the event role
        eventfillerlabel = None
        for fc, subj, pred, obj, objtype in coreconstraints:
            if subj == eventlabel and pred == eventrole:
                eventfillerlabel = obj
                break

        if eventfillerlabel is None:
            # not found
            return []

        # logging.info(f'event filler is {eventfillerlabel}')
        
        # find the filler of the object role
        entityfillerlabel = None
        entityrole = eventtype + "_" + self.paraphrasing_json["ComplexCause"]["EventTypes"][eventtype]["Entity"]
        for fc, subj, pred, obj, objtype in coreconstraints:
            if subj == eventlabel and pred == entityrole:
                entityfillerlabel = obj
                break

        if entityfillerlabel is None:
            # not found
            return []

        # logging.info(f'entity filler is {entityfillerlabel}')
        
        # is this filler of the event role also an event label, that is, is it actually an event?
        eventfillertype = None
        for fc, subj, pred, obj, objtype in coreconstraints:
            if subj == eventfillerlabel:
                # yes, found it
                eventfillertype = pred.split("_")[0]
                break

        if eventfillertype is None:
            # not found
            return []

        # logging.info(f'event filler is an event {eventfillertype}')
        
        # try to look up the agent role for this event type
        if eventfillertype in self.paraphrasing_json["ComplexCause"]["AgentRole"].keys():
            agentrole = eventfillertype + "_" + self.paraphrasing_json["ComplexCause"]["AgentRole"][eventfillertype]
        else:
            # not found
            return []

        # make the new constraint, if it is not already in there
        for fc, subj, pred, obj, objtype in coreconstraints:
            if fc == facetlabel and subj == eventfillerlabel and pred == agentrole and objtype == entityfillerlabel:
                # just the constraint we were going to make
                return [ ]
        new_constraint = [ facetlabel, eventfillerlabel, agentrole, entityfillerlabel, ""]

        # logging.info(f'successfully changed complex cause')
        
        # remove all occurrences of the old causal event, add the new constraint.
        # make a list of constraint sets, with just one member
        return [
            [ new_constraint ] + [(fc, subj, pred, obj, objtype) for (fc, subj, pred, obj, objtype) in coreconstraints\
                                         if subj != eventlabel]]

    # expand simple paraphrases:
    # returns a list of additional lists of core constraints
    def _expand_simple(self, coreconstraints):
        events_and_types = set( (fc, subj, pred.split("_")[0]) for fc, subj, pred, obj, objtype in coreconstraints)
        
        more_coreconstraint_sets = [ ]

        for facetlabel, eventlabel, eventtype in events_and_types:
            # are there simple paraphrases for this event?
            simple_paraphrases = self._has_simple_paraphrases(eventtype)
            for other_eventtype, rolemappings in simple_paraphrases.items():
                new_cc = [ ]
                for fc, subj, pred, obj, objtype in coreconstraints:
                    if fc == facetlabel and subj == eventlabel:
                        rolelabel = pred.split("_")[-1]
                        if rolelabel in rolemappings:
                            pred = other_eventtype + "_" + rolemappings[rolelabel]
                            new_cc.append( (fc, subj, pred, obj, objtype) )
                        else:
                            print("missing role label", pred, rolelabel)
                    else:
                        new_cc.append((fc, subj, pred, obj, objtype))
                            
                more_coreconstraint_sets.append(new_cc)


        return more_coreconstraint_sets

    
    # expand a given list of core constraints by parpharasing:
    # returns a list of additional s of core contraints
    def expand(self, coreconstraints):        
        events_and_types = set( (facet, subj, pred.split("_")[0]) for facet, subj, pred, obj, objtype in coreconstraints)

        # expand simple paraphrases
        more_coreconstraint_sets = self._expand_simple(coreconstraints)

        # expand complex cause paraphrases        
        for facetlabel, eventlabel, eventtype in events_and_types:
            if eventtype in self.paraphrasing_json["ComplexCause"]["EventTypes"].keys():
                new_ccs = self._complexcause(facetlabel, eventlabel, eventtype, coreconstraints)
                more_coreconstraint_sets += new_ccs
                for new_cc in new_ccs:
                    more_coreconstraint_sets += self._expand_simple(new_cc)

        return more_coreconstraint_sets


def testing():
    expansion_obj = HypothesisSeedExpansion()
    
    test_coreconstraints = [
        [ "FacetID", "?MiguelBrachoDeath", "Conflict.Attack_Attacker", "?MiguelBrachoKiller", ""],
        ["FacetID", "?MiguelBrachoDeath", "Conflict.Attack_Target", "?MiguelBracho", ""], 
        ["FacetID", "?MiguelBrachoDeath", "Confict.Attack__Place", "?Venezuela", ""]
        ]

    print("======== paraphrase from Conflict.Attack\n\n")
    additional = expansion_obj.expand(test_coreconstraints)
    for a in additional:
        print(a)
        print()

    
    test_coreconstraints = [
        [ "FacetID", "?MiguelBrachoDeath", "Life.Die.DeathCausedByViolentEvents_Killer", "?MiguelBrachoKiller", ""],
        ["FacetID", "?MiguelBrachoDeath", "Life.Die.DeathCausedByViolentEvents_Victim", "?MiguelBracho", ""], 
        ["FacetID", "?MiguelBrachoDeath", "Life.Die.DeathCausedByViolentEvents_Place", "?Venezuela", ""]
    ]

    print("======= paraphrase from Life.Die.DeathCausedByViolentEvents\n\n")
    additional = expansion_obj.expand(test_coreconstraints)
    for a in additional:
        print(a)
        print()
    

    test_coreconstraints = [
        ["FacetID", "?Violence", "Conflict.Attack_Place", "?Venezuela", ""],
        ["FacetID", "?ViolenceResponsibility", "ResponsibilityBlame.AssignBlame.AssignBlame_EntityResponsible",
             "?ViolenceEntityResponsible", ""],
        ["FacetID", "?ViolenceResponsibility", "ResponsibilityBlame.AssignBlame.AssignBlame_Event", "?Violence", ""]]

    print("========= first test, causal connection\n\n")
    additional = expansion_obj.expand(test_coreconstraints)
    for a in additional:
        print(a)
        print()

    test_coreconstraints = [
        ["FacetID", "?Demonstrations", "Conflict.Demonstrate.MarchProtestPoliticalGathering_Place", "?Venezuela", ""],
        ["FacetID", "?DemonstrationsSponsorship", "GeneralAffiliation.Sponsorship_ActorOrEvent", "?Demonstrations", ""],
        ["FacetID", "?DemonstrationsSponsorship", "GeneralAffiliation.Sponsorship_Sponsor", "?DemonstrationsSponsor", ""]]

    print("========= second test, causal connection\n\n")
    additional = expansion_obj.expand(test_coreconstraints)
    for a in additional:
        print(a)
        print()

    test_coreconstraints = [
        ["FacetID", "?DemonstrationsSponsorship", "GeneralAffiliation.Sponsorship_ActorOrEvent", "?DemonstrationActor", ""],
        ["FacetID", "?DemonstrationsSponsorship", "GeneralAffiliation.Sponsorship_Sponsor", "?DemonstrationsSponsor", ""]]

    print("========= third test, causal connection: should not yield changes\n\n")
    additional = expansion_obj.expand(test_coreconstraints)
    for a in additional:
        print(a)
        print()

        
# testing()
