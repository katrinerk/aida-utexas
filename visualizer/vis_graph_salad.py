import os
import dill
import json
import dash
from dash.dependencies import Input, Output
import dash_cytoscape as cyto
import dash_html_components as html
import dash_core_components as dcc
import sys

if len(sys.argv) == 1:
    file_name = 'test_ex/new_test_ext_orig.p'
else:
    file_name = sys.argv[1]

graph_dict = dill.load(open(file_name, 'rb'))
stmt_mat_ind = graph_dict['stmt_mat_ind']
query = {stmt_mat_ind.get_word(item) for item in graph_dict['query_stmts']}
graph_mix = graph_dict['graph_mix']
target_graph_id = graph_dict['target_graph_id']
query_eres = set.union(*[{graph_mix.stmts[stmt_id].head_id, graph_mix.stmts[stmt_id].tail_id} for stmt_id in query])
cand_stmts = {item for item in graph_mix.stmts.keys() if graph_mix.stmts[item].tail_id and set.intersection({graph_mix.stmts[item].head_id, graph_mix.stmts[item].tail_id}, query_eres) and item not in query}

for stmt_id in graph_mix.stmts.keys():
    if graph_mix.stmts[stmt_id].tail_id:
        assert len(set.intersection({graph_mix.stmts[stmt_id].head_id, graph_mix.stmts[stmt_id].tail_id}, set(graph_mix.eres.keys()))) == 2

for ere_id in graph_mix.eres.keys():
    temp = set.union(*[{graph_mix.stmts[stmt_id].head_id, graph_mix.stmts[stmt_id].tail_id} for stmt_id in graph_mix.eres[ere_id].stmt_ids if graph_mix.stmts[stmt_id].tail_id]) - {ere_id}
    assert temp == graph_mix.eres[ere_id].neighbor_ere_ids

    if graph_mix.eres[ere_id].category in ['Event', 'Relation']:
        for stmt_id in graph_mix.eres[ere_id].stmt_ids:
            if graph_mix.stmts[stmt_id].tail_id:
                assert graph_mix.stmts[stmt_id].head_id == ere_id
    else:
        for stmt_id in graph_mix.eres[ere_id].stmt_ids:
            if graph_mix.stmts[stmt_id].tail_id:
                assert graph_mix.stmts[stmt_id].tail_id == ere_id

mix_points = {ere_id for ere_id in graph_mix.eres.keys() if len({graph_mix.stmts[stmt_id].graph_id for stmt_id in graph_mix.eres[ere_id].stmt_ids}) == 3}
graph_ids = {graph_mix.stmts[stmt_id].graph_id for stmt_id in graph_mix.stmts.keys()}
graph_ids = [target_graph_id] + list(graph_ids - {target_graph_id})

elements = []

color_list = ['black', 'blue', 'orange']

for ere_id in graph_mix.eres.keys():
    ere = graph_mix.eres[ere_id]
    temp = dict()
    temp['data'] = dict()
    temp['data']['id'] = ere_id
    temp['data']['graph_id'] = ere.graph_id
    temp['data']['type'] = ere.category
    temp['data']['mix'] = 'Yes' if ere_id in mix_points else 'No'

    if ere.label and ere.category != "Relation":
        temp['data']['label'] = ere.label[0]
        temp['data']['length_label'] = max((len(ere.label[0]) * 10), 50)
    else:
        temp['data']['label'] = ''

    temp['data']['stmt_ids'] = list(ere.stmt_ids)

    elements.append(temp)

for stmt_id in graph_mix.stmts.keys():
    stmt = graph_mix.stmts[stmt_id]
    temp = {'data': dict()}
    temp['data']['id'] = stmt_id
    temp['data']['graph_id'] = stmt.graph_id
    temp['data']['label'] = stmt.raw_label
    temp['data']['source'] = stmt.head_id
    temp['data']['query'] = 'Yes' if stmt_id in query else 'No'
    temp['data']['cand'] = 'Yes' if stmt_id in cand_stmts else 'No'
    if stmt.tail_id:
        temp['data']['target'] = stmt.tail_id
    else:
        continue

    elements.append(temp)

print(elements[0])
app = dash.Dash(__name__)

default_stylesheet = [
    {
        'selector': 'node[type="Event"]',
        'style': {
            'shape': 'ellipse',
            'background-color': 'lightgreen',
            'border-style': 'solid',
            'border-width': '2',
            'label': 'data(label)',
            'width': 'data(length_label)',
            'height': '50',
            'text-halign': 'center',
            'text-valign': 'center'
        }
    },
    {
        'selector': 'node[type="Relation"]',
        'style': {
            'shape': 'circle',
            'background-color': 'blue',
            'border-style': 'solid',
            'border-width': '2',
            'width': '20',
            'height': '20',
        }
    },
    {
        'selector': 'node[type="Entity"]',
        'style': {
            'shape': 'ellipse',
            'background-color': 'yellow',
            'border-style': 'solid',
            'border-width': '2',
            'width': 'data(length_label)',
            'height': '50',
            'label': 'data(label)',
            'text-halign': 'center',
            'text-valign': 'center'
        }
    },
    {
        'selector': 'edge[graph_id="' + graph_ids[0] + '"]',
        'style': {
            'line-color': color_list[0],
            'curve-style': 'unbundled-bezier',
            'mid-target-arrow-shape': 'triangle',
            'mid-target-arrow-color': color_list[0],
            'arrow-scale': '1',
            'width': '1'
        }
    },
    {
        'selector': 'edge[graph_id="' + graph_ids[1] + '"]',
        'style': {
            'line-color': color_list[1],
            'curve-style': 'unbundled-bezier',
            'mid-target-arrow-shape': 'triangle',
            'mid-target-arrow-color': color_list[1],
            'arrow-scale': '1',
            'width': '1'
        }
    },
    {
        'selector': 'edge[graph_id="' + graph_ids[2] + '"]',
        'style': {
            'line-color': color_list[2],
            'curve-style': 'unbundled-bezier',
            'mid-target-arrow-shape': 'triangle',
            'mid-target-arrow-color': color_list[2],
            'arrow-scale': '1',
            'width': '1'
        }
    },
    {
        'selector': 'edge[query="Yes"]',
        'style': {
            'line-color': 'red',
            'mid-target-arrow-color': 'red'
        }
    },
    {
        'selector': 'edge[cand="Yes"]',
        'style': {
            'line-color': 'green',
            'mid-target-arrow-color': 'green'
        }
    },
    {
        'selector': 'node[mix="Yes"]',
        'style': {
            'background-color': 'lightblue'
        }
    },
]

app.layout = html.Div([
    cyto.Cytoscape(
        id='cytoscape',
        layout={'name': 'cose', 'nodeOverlap': '2500', 'randomize': 'false', 'nodeRepulsion': '1500'},
        elements=elements,
        style = {'width': '100%', 'height': '100%', 'position': 'absolute', 'top': '0px', 'left': '0px'},
    )
])

@app.callback(Output('cytoscape', 'stylesheet'),
              [Input('cytoscape', 'mouseoverEdgeData'),
               Input('cytoscape', 'mouseoverNodeData'),
               Input('cytoscape', 'tapNodeData')])
def generate_stylesheet(data_edge, data_node_hover, data_node_tap):
    if not (data_edge and data_node_hover and data_node_tap):
        return default_stylesheet

    stylesheet = default_stylesheet + [
{
        "selector": 'node[id = "{}"]'.format(data_node_tap['id']),
        "style": {
            'background-color': '#B10DC9',
            "label": "data(label)"
        }
    }]

    for edge in data_node_tap['stmt_ids']:
        stylesheet.append({
            "selector": 'edge[id = "{}"]'.format(edge),
            "style": {
                'background-color': '#B10DC9',
                "label": "data(label)",
                'text-border-width': 1,
                'text-background-color': 'yellow',
                'text-background-shape': 'rectangle',
                'text-background-opacity': 1,
                "text-border-style": 'solid',
                'text-border-opacity': 1,
                'text-halign': 'center',
                'text-valign': 'top'
            }
        })
    stylesheet.append(
        {
            "selector": 'edge[id = "{}"]'.format(data_edge['id']),
            "style": {
                'background-color': '#B10DC9',
                "label": "data(label)",
                'text-border-width': 1,
                'text-background-color': 'yellow',
                'text-background-shape': 'rectangle',
                'text-background-opacity': 1,
                "text-border-style": 'solid',
                'text-border-opacity': 1,
                'text-halign': 'center',
                'text-valign': 'top'
            }
        })

    label_list = [graph_mix.eres[data_node_hover['id']].label[0]] + ['+' * int(.75 * len(graph_mix.eres[data_node_hover['id']].label[0]))] + [('*' if graph_mix.stmts[stmt_id].graph_id == target_graph_id else '') + graph_mix.stmts[stmt_id].raw_label for stmt_id in graph_mix.eres[data_node_hover['id']].stmt_ids
                                                                                                                                              if not graph_mix.stmts[stmt_id].tail_id and stmt_id in query]
    label_list += ['-' * int(1.5 * len(graph_mix.eres[data_node_hover['id']].label[0]))] + [('*' if graph_mix.stmts[stmt_id].graph_id == target_graph_id else '') + graph_mix.stmts[stmt_id].raw_label for stmt_id in graph_mix.eres[data_node_hover['id']].stmt_ids
                                                                                                                                              if not graph_mix.stmts[stmt_id].tail_id and stmt_id not in query]
    stylesheet.append({
        "selector": 'node[id = "{}"]'.format(data_node_hover['id']),
        "style": {
                "label": ('\n').join(label_list),
                'text-wrap': 'wrap',
                'text-border-width': 1,
                'text-background-color': 'lightblue',
                'text-background-shape': 'rectangle',
                'text-background-opacity': 1,
                "text-border-style": 'solid',
                'text-border-opacity': 1,
                'text-halign': 'right',
                'text-valign': 'center'
        }
    })
    return stylesheet

if __name__ == '__main__':
    app.run_server(debug=True)