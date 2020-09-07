<----------Data Generation Pipeline---------->

-----Section 1: Graph Salads-----
Graph salads are artificial mixtures of topically-similar component graphs which a simulate a multi-narrative setting.
To create graph salads, we "collapse" two or more EREs from distinct source KGs into single representations.

Setup/Dependencies:
Run 'install_dependencies.sh' script via bash in this directory to install necessary Python packages
and download pretrained Word2Vec bin file.

The files required for graph salad data generation are as follows (in order):

****************************
(1) gen_single_doc_graphs.py
-------------------------------------------------------------------------------------------------------------------------
----Given a directory of single-doc json KG files, this script generates a directory containing each of those KGs
    represented as pickled Graph objects
-------------------------------------------------------------------------------------------------------------------------

****************************
(2) gen_event_entity_maps.py
-------------------------------------------------------------------------------------------------------------------------
----This script (i) takes in as input a directory containing single-doc graph instances (as pickled objects)
            and (ii) produces dictionaries of the form (key: event/entity name --> value: set of EREs with that name).
    This allows us to save time during data generation, as we can easily sample source graphs which have events/entities 
    with common names.
    
<Input/Output>
----Input signature: --graph_dir   ## Description: directory containing all single-doc KGs (as pickled objects)
                     --output_dir  ## Description: directory where name-to-ERE dicts will be written
----Output: Two name-to-ERE dicts (one for events, one for entities)
            Two ERE-to-type dicts (one for events, one for entities)
            Two ERE-to-connectedness dicts (one for one-step, one for two-step)
-------------------------------------------------------------------------------------------------------------------------

****************************
(3) gen_salads.py
-------------------------------------------------------------------------------------------------------------------------
----This script generates graph salads from a directory of single-doc KGs (as pickled objects).
     We create graph salads by mixing single-doc KGs at <num_shared_eres> event merge points.
     Once these event merge points have been identified, we then identify ALL entity EREs among the component graphs
     which (i) share a common name across the component graphs and (ii) are reachable from the event merge points.
     These entity EREs become additional merge points in the salad.

<Input/Output>
See gen_salads.py for a detailed list of arguments and their descriptions
-------------------------------------------------------------------------------------------------------------------------

****************************
(4) index_salads.py
-------------------------------------------------------------------------------------------------------------------------
----This script preps graph salads for training by indexing labels and constructing adjacency matrices.
-------------------------------------------------------------------------------------------------------------------------

----Input signature: --data_dir                  ## Description: directory containing Train/Val/Test data subdirectories
                     --indexer_dir               ## Description: directory where indexing/adjacency matrix info will be written
                     --emb_dim                   ## Description: dimension of pretrained Word2Vec embeds used
                     --return_frequency_cut_set: ## Description: number of tokens (most frequent) assigned a representation during training
-------------------------------------------------------------------------------------------------------------------------

