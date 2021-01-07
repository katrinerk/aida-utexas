# Katrin Erk December 2020
# first rought stab at query expansion for hypothesis seeds

import json
import logging
import uuid

class HypothesisSeedExpansion:
    def __init__(self):
        # hand-crafted paraphrases
        self.paraphrasing_json = {
            # simple paraphrase, symmetric:
            # just replace one event type by another,
            # with a mapping on arguments
            # We only specify this from the point of view of Conflict.Attack,
            # assuming that paraphrasing is symmetric and transitive.
            # Expansion to all paraphrase pairs needs to happen in the code
            "SimpleParaphraseSymmetric" : {
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
                        "Place": "Place"}
                    }
                },
            # simple paraphrase, asymmetric:
            # the key event type is a more general event type.
            # it can replace any of the given list of more specific event types,
            # in all cases using the same role mapping.
            # the mapping is expressed in the direction hypo -> hyper
            "SimpleParaphraseAsymmetric" : {
                "Conflict.Attack" : {
                    "Hypo" : ["Conflict.Attack.FirearmAttack", "Conflict.Attack.Bombing", "Conflict.Attack.AirstrikeMissileStrike",
                              "Conflict.Attack.BiologicalChemicalPoisonAttack", "Conflict.Attack.SelfDirectedBattle",
                              "Conflict.Attack.SetFire", "Conflict.Attack.Stabbing", "Conflict.Attack.Strangling"],
                    "Roles" : {
                        "Attacker": "Attacker",
                        "Target": "Target",
                        "Instrument": "Instrument",
                        "Place": "Place"}
                    },
                "Life.Die.DeathCausedByViolentEvents" : {
                    "Hypo" : ["Conflict.Attack.FirearmAttack", "Conflict.Attack.Bombing", "Conflict.Attack.AirstrikeMissileStrike",
                              "Conflict.Attack.BiologicalChemicalPoisonAttack", "Conflict.Attack.SelfDirectedBattle",
                              "Conflict.Attack.SetFire", "Conflict.Attack.Stabbing", "Conflict.Attack.Strangling"],
                    "Roles" : {
                        "Attacker": "Killer",
                        "Target": "Victim",
                        "Instrument": "Instrument",
                        "Place": "Place"}
                    },
                "Life.Injure" : {
                    "Hypo" : ["Conflict.Attack.FirearmAttack", "Conflict.Attack.Bombing", "Conflict.Attack.AirstrikeMissileStrike",
                              "Conflict.Attack.BiologicalChemicalPoisonAttack", "Conflict.Attack.SelfDirectedBattle",
                              "Conflict.Attack.SetFire", "Conflict.Attack.Stabbing", "Conflict.Attack.Strangling"],
                    "Roles" : {
                        "Attacker": "Injurer",
                        "Target": "Victim",
                        "Instrument": "Instrument",
                        "Place": "Place"}
                    },
                "Life.Injure.InjuryCausedByViolentEvents" : {
                    "Hypo" : ["Conflict.Attack.FirearmAttack", "Conflict.Attack.Bombing", "Conflict.Attack.AirstrikeMissileStrike",
                              "Conflict.Attack.BiologicalChemicalPoisonAttack", "Conflict.Attack.SelfDirectedBattle",
                              "Conflict.Attack.SetFire", "Conflict.Attack.Stabbing", "Conflict.Attack.Strangling"],
                    "Roles" : {
                        "Attacker": "Injurer",
                        "Target": "Victim",
                        "Instrument": "Instrument",
                        "Place": "Place"}
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
                        "Conflict.Coup.Coup" : "DeposingEntity"}},
            "PlaceRoles": [
                "Place", "Origin", "Destination", "HidingPlace"]
            }


    # internal function: given an event type,
    # identify other event types that work as paraphrases,
    # with role mappings
    def _has_simple_paraphrases(self, eventtarget):
        # mapping other_eventtype -> { myrolelabel -> otherrolelabel}
        retv = { }

        ##
        # work with symmetric paraphrases
        
        # there is an entry in the simple paraphrasing list where this event type is the head
        if eventtarget in self.paraphrasing_json["SimpleParaphraseSymmetric"]:
            retv.update(self.paraphrasing_json["SimpleParaphraseSymmetric"][eventtarget])

        # if there is an entry in th simple paraphrasing list where this event type is the tail:
        # rephrase paraphrasings to make this event type the head
        for eventhead in self.paraphrasing_json["SimpleParaphraseSymmetric"]:
            hpar = self.paraphrasing_json["SimpleParaphraseSymmetric"][eventhead]
            if eventtarget in hpar:
                # make entry for event head, with reversed role labels
                retv[ eventhead] = dict( (myrole, headrole) for headrole, myrole in hpar[eventtarget].items())
                # make entries for all the paraphrases of the event head, except myself
                for eventtype in hpar.keys():
                    if eventtype == eventtarget:
                        continue
                    retv[eventtype] = dict( (myrole, hpar[eventtype][headrole]) for myrole, headrole in retv[eventhead].items())

        ##
        # work with asymmetric paraphrases
        for eventhead in self.paraphrasing_json["SimpleParaphraseAsymmetric"]:
            if eventtarget in self.paraphrasing_json["SimpleParaphraseAsymmetric"][eventhead]["Hypo"]:
                # the event target can be replaced by a more general paraphrase that is eventhead.
                retv[eventhead] = self.paraphrasing_json["SimpleParaphraseAsymmetric"][eventhead]["Roles"]
        return retv

    # given an event type that could signal a complex cause,
    # check whether we really have a match.
    # if so, make a paraphrase
    def _complexcause(self, eventlabel, eventtype, coreconstraints):
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
        facetlabel = None
        entityrole = eventtype + "_" + self.paraphrasing_json["ComplexCause"]["EventTypes"][eventtype]["Entity"]
        for fc, subj, pred, obj, objtype in coreconstraints:
            if subj == eventlabel and pred == entityrole:
                entityfillerlabel = obj
                facetlabel = fc
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
            if subj == eventfillerlabel and pred == agentrole and objtype == entityfillerlabel:
                # just the constraint we were going to make
                return [ ]
        new_constraint = [ facetlabel, eventfillerlabel, agentrole, entityfillerlabel, None]

        # logging.info(f'successfully changed complex cause')
        
        # remove all occurrences of the old causal event, add the new constraint.
        # make a list of constraint sets, with just one member
        return [
            [ new_constraint ] + [(fc, subj, pred, obj, objtype) for (fc, subj, pred, obj, objtype) in coreconstraints\
                                         if subj != eventlabel]]

    # expand simple paraphrases:
    # returns a list of additional lists of core constraints
    def _expand_simple(self, coreconstraints):
        events_and_types = set( (subj, pred.split("_")[0]) for fc, subj, pred, obj, objtype in coreconstraints)
        
        more_coreconstraint_sets = [ ]

        for eventlabel, eventtype in events_and_types:
            # are there simple paraphrases for this event?
            simple_paraphrases = self._has_simple_paraphrases(eventtype)
            for other_eventtype, rolemappings in simple_paraphrases.items():
                new_cc = [ ]
                for fc, subj, pred, obj, objtype in coreconstraints:
                    if subj == eventlabel:
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

    # expand temporal paraphrases:
    # if:
    # Event --Place--> X
    #       --[participant that is not a place]--> Y
    #       --Time--> T
    # then
    # LocatedNear --Place--> X
    #             --EntityOrFiller--> Y
    #             -- Time --> T
    def _expand_temporal(self, coreconstraints, tempconstraints, entrypoints):
        # keep subj for cases where subj is an event label that has a temporal constraint
        tempevents = set(subj for facet, subj, pred, obj, objtype in coreconstraints if subj in tempconstraints.keys())

        more_coreconstraint_sets = [ ]
        more_tempconstraints = { }
        
        for eventlabel in tempevents:
            thisplace = None
            thisplacefacet = None
            this_nonplaceargs = set()
            for fc, subj, pred, obj, objtype in coreconstraints:
                if subj == eventlabel:
                    role = pred.split("_")[-1]
                    if obj in entrypoints: 
                        if role in self.paraphrasing_json["PlaceRoles"]:
                            thisplace = obj
                            thisplacefacet = fc
                        else:
                            this_nonplaceargs.add((fc, obj))

            # did we get a place argument that is an entry point? then we can make the paraphrase
            if thisplace is not None and len(this_nonplaceargs) > 0:
                new_cc = [ ]
                new_evlabel = "LocatedEvt" + str(uuid.uuid4())
                new_cc.append((thisplacefacet, new_evlabel, "Physical.LocatedNear_Place", thisplace, None))
                for facetlabel, participant in this_nonplaceargs:
                    new_cc.append((facetlabel, new_evlabel, "Physical.LocatedNear_EntityOrFiller", participant, None))
                    
                more_coreconstraint_sets.append(new_cc)
                more_tempconstraints[new_evlabel] = tempconstraints[eventlabel]

        return (more_coreconstraint_sets, more_tempconstraints)
                    
                    
    # expand a given list of core constraints by parpharasing:
    # returns a list of additional s of core contraints
    def expand(self, coreconstraints, temporalconstraints, entrypoints):        
        events_and_types = set( (subj, pred.split("_")[0]) for facet, subj, pred, obj, objtype in coreconstraints)

        more_coreconstraint_sets = [ ]
        
        # expand simple paraphrases
        more_coreconstraint_sets = self._expand_simple(coreconstraints)

        # expand complex cause paraphrases        
        for eventlabel, eventtype in events_and_types:
            if eventtype in self.paraphrasing_json["ComplexCause"]["EventTypes"].keys():
                new_ccs = self._complexcause(eventlabel, eventtype, coreconstraints)
                more_coreconstraint_sets += new_ccs
                for new_cc in new_ccs:
                    more_coreconstraint_sets += self._expand_simple(new_cc)

        # expand temporal paraphrases
        temp_paraphrases, more_temporalconstraints = self._expand_temporal(coreconstraints, temporalconstraints, entrypoints)
        more_coreconstraint_sets += temp_paraphrases
                    
        return (more_coreconstraint_sets, more_temporalconstraints)


def testing():
    expansion_obj = HypothesisSeedExpansion()
    
    test_coreconstraints = [
        [ "FacetID", "?MiguelBrachoDeath", "Conflict.Attack_Attacker", "?MiguelBrachoKiller", ""],
        ["FacetID", "?MiguelBrachoDeath", "Conflict.Attack_Target", "?MiguelBracho", ""], 
        ["FacetID", "?MiguelBrachoDeath", "Confict.Attack_Place", "?Venezuela", ""]
        ]

    print("======== paraphrase from Conflict.Attack\n\n")
    additional, _ = expansion_obj.expand(test_coreconstraints, {}, ["?MiguelBracho", "?Venezuela"])
    for a in additional:
        print(a)
        print()

    
    test_coreconstraints = [
        [ "FacetID", "?MiguelBrachoDeath", "Life.Die.DeathCausedByViolentEvents_Killer", "?MiguelBrachoKiller", ""],
        ["FacetID", "?MiguelBrachoDeath", "Life.Die.DeathCausedByViolentEvents_Victim", "?MiguelBracho", ""], 
        ["FacetID", "?MiguelBrachoDeath", "Life.Die.DeathCausedByViolentEvents_Place", "?Venezuela", ""]
    ]

    print("======= paraphrase from Life.Die.DeathCausedByViolentEvents\n\n")
    additional, _ = expansion_obj.expand(test_coreconstraints, {}, ["?MiguelBracho", "?Venezuela"])
    for a in additional:
        print(a)
        print()
    
    test_coreconstraints = [
        [ "FacetID", "?MiguelBrachoDeath", "Conflict.Attack.Bombing_Attacker", "?MiguelBrachoKiller", ""],
        ["FacetID", "?MiguelBrachoDeath", "Conflict.Attack.Bombing_Target", "?MiguelBracho", ""], 
        ["FacetID", "?MiguelBrachoDeath", "Conflict.Attack.Bombing_Place", "?Venezuela", ""]
    ]

    print("======= paraphrase from Conflict.Attack.Bombing\n\n")
    additional, _ = expansion_obj.expand(test_coreconstraints, {}, ["?MiguelBracho", "?Venezuela"])
    for a in additional:
        print(a)
        print()

        
    test_coreconstraints = [
        ["FacetID", "?Violence", "Conflict.Attack_Place", "?Venezuela", ""],
        ["FacetID", "?ViolenceResponsibility", "ResponsibilityBlame.AssignBlame.AssignBlame_EntityResponsible",
             "?ViolenceEntityResponsible", ""],
        ["FacetID", "?ViolenceResponsibility", "ResponsibilityBlame.AssignBlame.AssignBlame_Event", "?Violence", ""]]

    print("========= first test, causal connection\n\n")
    additional, _ = expansion_obj.expand(test_coreconstraints, {}, ["?Venezuela"])
    for a in additional:
        print(a)
        print()

    test_coreconstraints = [
        ["FacetID", "?Demonstrations", "Conflict.Demonstrate.MarchProtestPoliticalGathering_Place", "?Venezuela", ""],
        ["FacetID", "?DemonstrationsSponsorship", "GeneralAffiliation.Sponsorship_ActorOrEvent", "?Demonstrations", ""],
        ["FacetID", "?DemonstrationsSponsorship", "GeneralAffiliation.Sponsorship_Sponsor", "?DemonstrationsSponsor", ""]]

    print("========= second test, causal connection\n\n")
    additional, _ = expansion_obj.expand(test_coreconstraints, {}, ["?Venezuela"])
    for a in additional:
        print(a)
        print()

    test_coreconstraints = [
        ["FacetID", "?DemonstrationsSponsorship", "GeneralAffiliation.Sponsorship_ActorOrEvent", "?DemonstrationActor", ""],
        ["FacetID", "?DemonstrationsSponsorship", "GeneralAffiliation.Sponsorship_Sponsor", "?DemonstrationsSponsor", ""]]

    print("========= third test, causal connection: should not yield changes\n\n")
    additional, _ = expansion_obj.expand(test_coreconstraints, {}, [])
    for a in additional:
        print(a)
        print()

    test_coreconstraints = [
        [ "FacetID", "?MiguelBrachoDeath", "Conflict.Attack_Attacker", "?MiguelBrachoKiller", ""],
        ["FacetID", "?MiguelBrachoDeath", "Conflict.Attack_Target", "?MiguelBracho", ""], 
        ["FacetID", "?MiguelBrachoDeath", "Confict.Attack_Place", "?Venezuela", ""]
        ]

    print("========== testing emporal constraints ========")
    test_tempconstraints = { "?MiguelBrachoDeath" :
                                 { "start_time" : "2018-01-01",
                                    "end_time": "2018-01-01"}}

    additional, newtemp = expansion_obj.expand(test_coreconstraints, test_tempconstraints, ["?MiguelBracho", "?Venezuela"])
    for a in additional:
        print(a)
        print()

    for qvar, keyval in newtemp.items():
        for key, val in keyval.items():
            print("temporal:", qvar, key, val)
    
# testing()
