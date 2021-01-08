README for the use of analyze_hypotheses.py

The tool reads in a TA2 KB, along with result jsons from a run of our system,
and displays information about the TA2 KB and the hypotheses we computed
for manual analysis.

After starting up, the tool goes into interactive mode and displays a menu of possible analyses.

------------------------------------
Usage:


python3 analyze_hypotheses.py <graph_json_filename> <hypothesis_file> <roles_ontology_path>

for example:
analyze_hypotheses.py ~/Documents/Projects/AIDA/data/mytestdir/outdir/WORKING/GaiaOperaColorado/GaiaOperaColorado.json ~/Documents/Projects/AIDA/data/mytestdir/outdir/WORKING/GaiaOperaColorado/result_jsons_filtered/T203.json
~/Documents/Projects/AIDA/scripts/aida-utexas/resources/roles_ontology_phase2_v1.json 
------------------------------------
Overview of menu options:

c: core hypothesis display:
    Displays, on the screen, all core hypothesis statements across all hypotheses in the hypothesis set.
    statements that are about the same event type/ relation type are pooled such that you see all fillers of the same role in the same
    core event/relation listed one after the other
    
    Hypothesis display can optionally be restricted to hypotheses with a particular entity ID in a particular core role slot,
    the system asks about this. Example below.

e: show events/relations connected to an ere:
    For an event, relation or entity that appears in at least one core hypothesis, show all adjacent events and relations.
    If the ERE is an event or relation, this just shows that event or relation with all fillers.
    If the ERE is an entity, shows all event/relations mentioned in any of the hypotheses (not just core) that is about the ERE

    Hypothesis display can optionally be restricted to hypotheses with a particular entity ID in a particular core role slot,
    the system asks about this. Example below.


r: show events/relations connected to a role filler:
    For an event or relation that appears in some core hypotheses, and a role of that event/relation, show all entities that fill that role
    in any of the hypotheses, along with all other events and relations that are mentioned in any hypothesis about that entity
    
    Hypothesis display can optionally be restricted to hypotheses with a particular entity ID in a particular core role slot,
    the system asks about this. Example below.

se: survey context of an ERE independent of hypotheses:
    Like option e, but also shows events and relations that are not in any of the hypotheses but are in the KB.
    This answers the question: What *else* could our system possibly have found out about the ERE of interest
    in the graph.

sr: survey context of a role filler independent of hypotheses:
    Like option r, but also shows events and relations that are not in any of the hypotheses but are in the KB.
    
p: print hypotheses for a particular question ID:
     Print, to a given directory, a human-readable version of all hypotheses for a given question ID.
     This is similar to our normal human-readable text format, but it first lists all core events and relations of
     a hypothesis, then, for each entity mentioned in the hypothesis, all events and relations that mention that entity.
     So it takes a more entity-centric view.

    Hypothesis display can optionally be restricted to hypotheses with a particular entity ID in a particular core role slot,
    the system asks about this. Example below.


R: restrict hypotheses to be considered going forward, for the rest of the run:
    For the rest of the run (until you hit x), only consider hypotheses that have a particular core event/relation role filled
    by a particular ERE ID.  This is like the optional restriction of hypotheses above, but while the optional restrictions above are only
    kept for the execution of that one menu choice, restrictions made with R stay around for all subsequent menu choices. 

x: exit


------------------------------------
Some sample runs. for the dry run SIN

-------
Overall, what did we find for question Q002 (what destroyed the drones)?

* menu choice: c: display core hypotheses 
* system asks: Question ID
  user says: Q002   (that is, display all hypotheses that have Q002 as a substring in their ID)
  system asks: Optional core role to restrict: (use this to only display hypotheses that have a particular entity ID in a particular core role)
  user says: [enter] (no further restriction)
  system shows: all core events of SIN with question ID Q002, with a list of entities, across all hypotheses, that filled it, including:

  ArtifactExistence.DamageDestroy.Destroy_Artifact
  	 Constituent National Assembly,Communist Party of Chile,Asamblea General
		 http://www.isi.edu/gaia/entities/uiuc/KC003AE11/EN_Entity_EDL_0006795
	 FAC
		 http://www.isi.edu/gaia/entities/uiuc/KC003AE23/EN_Entity_EDL_0030378
	 Колумбия,Aragua Area,colombiano
		 http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/entity-instance-KC003AE11-r202010170007-794
	 FAC.Structure.Barricade
		 http://www.isi.edu/gaia/entities/uiuc/KC003AE3M/EN_Entity_EDL_0015099
  ArtifactExistence.DamageDestroy.Destroy_Destroyer
	 Su,éste,La saña
		 http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/entity-instance-KC003AE11-r202010170007-802
	 PER.MilitaryPersonnel
		 http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/entity-instance-KC003AE23-r202010170014-96
  [...]

--------
What did we find for question Q003 (who was behind the drone attack) if we only consider attack instruments that were actually drones?

* menu choice c: display core hypotheses
* system asks: Question ID
  user says: Q003   (that is, display all hypotheses that have Q003 as a substring in their ID)
  system asks: Optional core role to restrict: (use this to only display hypotheses that have a particular entity ID in a particular core role)
  user says: [enter] (no further restriction)
  system shows: all core events of SIN with question ID Q003, with a list of entities, across all hypotheses, that filled it, including:

  Conflict.Attack_Instrument
	 WEA
		 http://www.isi.edu/gaia/entities/uiuc/IC001VBFL/EN_Entity_EDL_0026335
		 http://www.isi.edu/gaia/entities/uiuc/IC001VG5D/RU_Entity_EDL_0005606
		 http://www.isi.edu/gaia/entities/uiuc/JC002YBKV/EN_Entity_EDL_0011520
		 http://www.isi.edu/gaia/entities/uiuc/IC001VG14/EN_Entity_EDL_0021717
	 VEH.Aircraft.Drone
		 http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/entity-instance-IC001VBJS-r202010162352-30
		 http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/entity-instance-IC001VBFL-r202010162342-6
		 http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/entity-instance-IC001VBFL-r202010162342-41
		 http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/entity-instance-IC001VBJC-r202010170019-165
  [...]

* menu choice c: display core hypotheses
* system asks: Question ID
* user says: Q003
  system asks: Optional core role to restrict:
  user says: Conflict.Attack_Instrument (that is, we only want hypotheses with a particular value for the instrument of attack)
  system asks: Value to restrict the core role to (ERE ID):
  user says: http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/entity-instance-IC001VBJS-r202010162352-30
     (copying down from above one of the entity IDs of an entity that actually seems to be a drone)

* system says:
  Conflict.Attack_Instrument
	 VEH.Aircraft.Drone
		 http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/entity-instance-IC001VBJS-r202010162352-13
		 http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/entity-instance-IC001VBJS-r202010162352-30
  Disaster.FireExplosion.FireExplosion_FireExplosionObject
	 FAC.Building
		 http://www.isi.edu/gaia/entities/uiuc/IC001VBJS/EN_Entity_EDL_0027294
  Conflict.Attack_Place
	 FAC.Building
		 http://www.isi.edu/gaia/entities/uiuc/IC001VBJS/EN_Entity_EDL_0027294
	 Колумбия,Aragua Area,colombiano
		 http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/entity-instance-KC003AE11-r202010170007-794
  Conflict.Attack_Attacker
	 GPE
		 http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/entity-instance-IC001VBJS-r202010162352-60
  Conflict.Attack_Target
	 Vladimir Padrino,Nicolas MADURO Moros,quien
		 http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/entity-instance-KC003AE11-r202010170007-953

( here we would have to repeat this for every entity ID that is listed as an actual drone)

--------
For question Q003 (who was behind the drone attack) we just got an answer with a particular entity ID. Who is that?

* menu choice:  e (display info that hypotheses have about an entity)
* system says: Question ID
* user says: Q003
* system says: Optional core role to restrict:
* user says: [enter] (no further restrictions; here we could also again have restricted to hypotheses where the Conflict.Attack_Instrument has a particular ID
                               that we know to be the ID of a drone)
* system says: ERE ID to inspect:
* user says: http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/entity-instance-IC001VBJS-r202010162352-60 (that is the attacker from above)
* system says:
  =============

  ERE is Conflict.Attack_Attacker in:
  Conflict.Attack
	  Attacker: GPE 202010162352-60
	  Target: se 02010170007-953
	   Instrument: VEH.Aircraft.Drone 202010162352-13
	   	   VEH.Aircraft.Drone 202010162352-30
    	  Place: FAC.Building ity_EDL_0027294
	  	 u 02010170007-794

  ERE is ResponsibilityBlame.ClaimResponsibility.ClaimResponsibility_EntityResponsible in:
   ResponsibilityBlame.ClaimResponsibility.ClaimResponsibility
      EntityResponsible: GPE 202010162352-60
      Event: Conflict.Attack EN_Event_006643

  ERE is Movement.TransportPerson.SelfMotion_Transporter,Movement.TransportArtifact_Artifact in:
    Movement.TransportPerson.SelfMotion
      Transporter: GPE 202010162352-60
      Vehicle: VEH.Aircraft.Drone 202010162352-13
      Origin: 
      Destination: 

------------
For that same entity ID, what does the KB know that may not be in any hypothesis?

* menu choice: se (display KB info about an entity)
* system says: ERE ID to inspect:
* user says:  http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/entity-instance-IC001VBJS-r202010162352-60 (same as above)
* system says: [same answer as above] (so there isn't anything the KB knows about this entity that is not in our hypotheses)

---------------
Ignoring the problem that many instruments of attack in Q003 hypotheses don't seem to be drones,
what do we know about *all* attackers in this core attack event, as listed in the hypotheses?

* menu choice: r
* system says: Question ID:
* user says: Q003
* system says: Optional core role to restrict:
* user says: [enter]
* system says: Role label to inspect:
* user says: Conflict.Attack_Attacker (this looks at all entities that fill the attacker role in
     the attack event that is core for this question)
* system says:
   [ for each entity that appears as attacker in some hypothesis,
    a list of everything that any hypothesis has to say about that entity,
    with statements from different hypotheses separated by '==========='
]
