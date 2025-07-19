from collections import defaultdict, deque
from typing import Dict, List
import json

from utils import init_minimal_transitions

import networkx as nx
from pyvis.network import Network
import matplotlib as mpl


class HasseDiagram(nx.Graph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)


    @classmethod
    def generate_from_quiver(cls, starting_quiver):
        '''
        Build a graph of quivers reachable from `starting_quiver` via minimal transitions.

        Nodes: canonical representative quiver objects (unique up to equivalence).
        Edges: transitions between quivers.
        Each node has attribute 'dimension' = Coulomb dimension.
        '''
        graph = cls()
        dimension_upper_bound = starting_quiver.get_coulomb_dimension()
        minimal_transitions = init_minimal_transitions(dimension_upper_bound)

        # Keep representatives by Coulomb dimension for equivalence checking
        reps_by_dim: Dict[int, List] = defaultdict(list)

        def get_rep(quiver):
            """
            Return the representative for `quiver`, adding a new one if none equivalent exists.
            """
            dim = quiver.get_coulomb_dimension()
            for rep in reps_by_dim[dim]:
                if quiver.is_equivalent(rep):
                    return rep
            # no equivalent found => treat as new representative
            reps_by_dim[dim].append(quiver)
            return quiver

        # BFS over quivers
        queue = deque([starting_quiver])
        visited = set()

        while queue:
            q = queue.popleft()
            rep_q = get_rep(q)

            # ensure node exists
            if rep_q not in graph:
                graph.add_node(rep_q, dimension=rep_q.get_coulomb_dimension())

            # explore all minimal transitions
            for mt in minimal_transitions:
                # subtract may return list of new quivers or []
                for new_q in q.subtract(mt):
                    rep_new = get_rep(new_q)

                    # add new node if missing
                    if rep_new not in graph:
                        graph.add_node(rep_new, dimension=rep_new.get_coulomb_dimension())

                    # add edge
                    if not graph.has_edge(rep_q, rep_new):
                        graph.add_edge(rep_q, rep_new, transition=mt.name)

                    # queue for further exploration
                    if rep_new not in visited:
                        visited.add(rep_new)
                        queue.append(new_q)

        return graph

    def plot_html(self, path):

        # quiver to id in graph
        mapping = {quiver: i for i, quiver in enumerate(self.nodes())}
        hasse = nx.relabel_nodes(self, mapping)

        # skipping 'empty' dimensions in plot
        dimensions = []
        for node in hasse.nodes():
            dimension = hasse.nodes[node]['dimension']

            if dimension not in dimensions:
                dimensions.append(dimension)

        dimensions = sorted(dimensions)
        dimension_to_level = {dimension: i for i,dimension in enumerate(dimensions)}

        # assigning colors to different transitions
        transitions = []
        for edge in hasse.edges():
            curr_transition = hasse.edges[edge]['transition']

            if curr_transition not in transitions:
                transitions.append(curr_transition)

        number_transitions = len(transitions)
        cmap = mpl.colormaps['tab20']
        transition_to_color = {}
        for i, curr_transition in enumerate(transitions):
            rgba = cmap(i / max(number_transitions - 1, 1))
            hex_color = mpl.colors.to_hex(rgba)
            transition_to_color[curr_transition] = hex_color

        # pyvis network
        network = Network(
            height='800px',
            width='100%',
            directed=False,
            notebook=False,
            cdn_resources='remote',
        )
        network.from_nx(hasse)

        # assigning node attributes
        for node in network.nodes:
            nid = node['id']
            dim = hasse.nodes[nid]['dimension']
            node['level'] = dimension_to_level[dim]
            node['label'] = f"{dim}"
            node['title'] = f"Dimension = {dim}"

        # finding unique transitions
        for edge in network.edges:
            u = edge['from']
            v = edge['to']

            transition = hasse.edges[(u,v)]['transition']
            edge['label'] = transition
            edge['title'] = f"Transitions: {transition}"
            edge['color'] = transition_to_color[transition]

        # 8) Set hierarchical layout options, show options in browser

        opts = {
            "layout": {
                "hierarchical": {
                "enabled": True,
                "direction": "DU",
                "levelSeparation": 150,
                "nodeSpacing": 100,
                "sortMethod": "directed",
                "edgeColor": {
                    "inherit": False
                }
                }
            },
            "physics": { 
                "enabled": True 
            },
            "configure": {
                "enabled": True,
                "filter": ["physics"]      # <-- this is exactly what show_buttons(filter_=["physics"]) would have done
            }
        }
        network.set_options(json.dumps(opts))
        #network.show_buttons(filter_=["physics"])

        #network.show(path)
        network.write_html(path, open_browser=True, notebook=False)