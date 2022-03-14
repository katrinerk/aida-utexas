from aida_utexas.aif.aida_graph import AidaGraph, AidaNode
from aida_utexas.aif.json_graph import JsonGraph
from aida_utexas.aif.rdf_graph import RDFGraph, RDFNode

AIF_NS_PREFIX = 'https://raw.githubusercontent.com/NextCenturyCorporation/' \
                'AIDA-Interchange-Format/master/java/src/main/resources/com/ncc/aif/ontologies'
                
AIDA = f'<{AIF_NS_PREFIX}/InterchangeOntology#>'
LDC = f'<{AIF_NS_PREFIX}/LdcAnnotations#>'
LDC_ONT = f'<{AIF_NS_PREFIX}/LDCOntologyM36#>'
UTEXAS = '<http://www.utexas.edu/aida/>'
