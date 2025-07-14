import networkx as nx
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pyvis.network import Network


class Quiver:
    def __init__(self, nodes: list, edges: list, node_values: list, name: str = None):
        '''Initialize a Quiver with edges and node values.
        
        Args:
            nodes (list): List of node ids.
            edges (list): List of tuples representing edges between nodes.
            node_values (list): List of values for each node.
        '''

        self.quiver = nx.MultiGraph()
        self.quiver.add_nodes_from(nodes)
        self.quiver.add_edges_from(edges)

        #assert set(nodes) == set(self.quiver.nodes()), 'Edges must be compatible with nodes'
        assert len(nodes) == len(node_values), "Node values must match the number of nodes."

        for node, value in zip(nodes, node_values):
            self.quiver.nodes[node]['value'] = value

        self.name = name


    def find_all_embeddings(self, sub_quiver) -> list[dict]:
        '''
        Find all subgraph isomorphic embeddings of 'sub_quiver' into this quiver.

        Args:
            sub_quiver (Quiver): The subgraph to find embeddings for.

        Returns:
            List[dict]: Each dict maps sub-quiver → quiver node for one embedding.
        '''
        G = self.quiver
        H = sub_quiver.quiver

        if H.number_of_nodes() > G.number_of_nodes() or H.number_of_edges() > G.number_of_edges():
            return []

        gm = isomorphism.GraphMatcher(G, H)
        mappings = []

        for mapping_g2h in gm.subgraph_isomorphisms_iter():
            # invert G→H into H→G
            mapping_h2g = {h: g for g, h in mapping_g2h.items()}
            # optional dedup: skip if same set of target nodes already seen
            node_set = frozenset(mapping_h2g.values())
            if all(node_set != frozenset(m.values()) for m in mappings):
                mappings.append(mapping_h2g)

        if mappings:
            return mappings
        else:
            return []

    def is_equivalent(self, other_quiver):
        graph1 = self.quiver
        graph2 = other_quiver.quiver

        if set(graph1.nodes()) != set(graph2.nodes()): return False
        
        for node in graph1.nodes():
            if graph1.nodes[node] != graph2.nodes[node]: return False

        if set(graph1.edges()) != set(graph2.edges()): return False

        return True

    def subtract(self, sub_quiver):
        '''
        Subtract the values of the nodes in 'sub_quiver' from this quiver. If sub_quiver is not found in this quiver, return the original quiver.

        Args:
            sub_quiver (Quiver): The subgraph to subtract.

        Returns:
            Quiver: A new Quiver with the values subtracted.
        '''

        mappings = self.find_all_embeddings(sub_quiver)
        if not mappings: return []# No embeddings found, return original quiver

        results = []
        subgraph = sub_quiver.quiver
        for mapping in mappings:
            graph = self.quiver.copy() # New quiver for each mapping
            old_balance = self.get_balance()

            for h_node, g_node in mapping.items():
                graph.nodes[g_node]['value'] -= subgraph.nodes[h_node]['value']

                if graph.nodes[g_node]['value'] <= 0:
                    graph.remove_node(g_node)

            new_quiver = Quiver.from_graph(graph, name=f"{self.name} - {sub_quiver.name}")
            new_quiver = new_quiver.restore_balance(old_balance)
            results.append(new_quiver)

        if results:
            return results
        else:
            return []


    def restore_balance(self, old_balance):
        '''
        Adds an overall U(1) node to the quiver and adds edges to restore the balance of nodes back to their pre-subtraction
        values.
        '''
        graph = self.quiver.copy()
        curr_balance = self.get_balance()

        additional_node = max(graph.nodes()) + 1 if graph.nodes() else 1
        graph.add_node(additional_node)
        graph.nodes[additional_node]['value'] = 1

        for node, bal in curr_balance.items():
            if bal == old_balance[node]: continue
            if bal > old_balance[node]: print('Dont think this should happen..')

            num_additional_edges = old_balance[node] - bal

            graph.add_edges_from([
                (node, additional_node) for _ in range(num_additional_edges)
            ])

        return Quiver.from_graph(graph, name=self.name)


    def display(self):
        '''Display the multi graph quiver using matplotlib.'''
        quiver = self.quiver
        #pos = nx.spring_layout(quiver, seed=42)
        #pos = nx.planar_layout(quiver)
        #pos = nx.bfs_layout(quiver, 1)
        pos = nx.kamada_kawai_layout(quiver)
        #pos = nx.multipartite_layout(quiver)

        # draw nodes and labels
        nx.draw_networkx_nodes(quiver, pos, node_color="lightblue", node_size=400)

        # extract ids and offset positions
        node_ids = list(quiver.nodes())
        labels_ids = {node: f"id: {node}" for node in node_ids}
        labels_ids_pos = {node: (pos[node][0], pos[node][1] + 0.1) for node in node_ids}
        nx.draw_networkx_labels(quiver, labels_ids_pos, labels=labels_ids, font_size=8, verticalalignment='bottom')

        labels_values = {node: f"{quiver.nodes[node]['value']}" for node in node_ids}
        nx.draw_networkx_labels(quiver, pos, labels=labels_values, font_size=8)

        # draw each parallel edge with a different curvature
        # group edges by (u,v) ignoring order
        seen = {}
        for u, v, key in quiver.edges(keys=True):
            # determine how many times we've already drawn an edge between u,v
            pair = tuple(sorted((u, v)))
            i = seen.get(pair, 0)
            seen[pair] = i + 1

            # radial offset: spread them +/- around a small radius
            # if there are N parallel edges between u and v, i goes 0..N-1
            N = quiver.number_of_edges(u, v)
            # center them about zero
            rad = (i - (N-1)/2) * 0.2

            nx.draw_networkx_edges(
                quiver, pos,
                edgelist=[(u, v)],
                connectionstyle=f"arc3,rad={rad}",
                edge_color="gray"
            )

        plt.title(self.name if self.name else "Quiver") 
        plt.axis('off')
        plt.show()


    def get_balance(self):
        balance = {}
        graph = self.quiver
        for node in graph.nodes():

            connected_value_sum = 0
            for neighbor in graph.neighbors(node):
                connected_value_sum += graph.nodes[neighbor]['value'] * graph.number_of_edges(node, neighbor)

            balance[node] = connected_value_sum - 2 * graph.nodes[node]['value']

        return balance


    def get_coulomb_dimension(self):
        graph = self.quiver
        dimension = 0
        for node in graph.nodes():
            dimension += graph.nodes[node]['value']

        return dimension - 1 # minus 1 corresponds to factorign out U(1)


    @classmethod
    def from_graph(cls, graph: nx.MultiGraph, name: str = None) -> 'Quiver':
        '''
        Create a Quiver from a NetworkX MultiGraph.

        Args:
            graph (nx.MultiGraph): The graph to convert.
            name (str): Optional name for the quiver.

        Returns:
            Quiver: A new Quiver instance.
        '''
        nodes = []
        node_values = []
        for node, data in graph.nodes(data=True):
            nodes.append(node)
            node_values.append(data['value'])

        edges = list(graph.edges(keys=True))

        return Quiver(nodes, edges, node_values, name=name)


def init_minimal_transitions(dim_upper_bound: int = 10):
    '''Some minimal nilpotent orbit transitions... CLEAN THIS, INDICES ARE TOO MESY'''

    an_quivers = []
    an_upper_bound = dim_upper_bound
    for n in range(1, an_upper_bound + 1):
        nodes = [i+1 for i in range(n+1)]
        linear_edges = [(i, i+1) for i in range(1, n)]
        all_edges = linear_edges + [(n, n+1)] + [(n+1, 1)]
        values = [1] * (n + 1)

        an_quivers.append(
            Quiver(nodes, all_edges, values, name=f"a_{n}")
        )

    dn_quivers = []
    #dn_upper_bound = (dim_upper_bound + 3) / 2 if (dim_upper_bound + 3) % 2 == 0 else (dim_upper_bound + 2) / 2
    dn_upper_bound = 5
    for n in range(4, dn_upper_bound + 1):
        nodes = [i+1 for i in range(n+1)]
        linear_edges = [(i, i+1) for i in range(1, n-1)]
        all_edges = linear_edges + [(2, n)] + [(n-2, n+1)]
        values = [1] + [2] * (n - 3) + [1] + [1, 1]

        dn_quivers.append(
            Quiver(nodes, all_edges, values, name=f"d_{n}")
        )

    An_quivers = []
    An_upper_bound = dim_upper_bound
    for n in range(2, An_upper_bound+1):
        nodes = [1, 2]
        edges = [(1,2) for _ in range(n)]
        values = [1, 1]

        An_quivers.append(
            Quiver(nodes, edges, values, name=f'A_{n}')
        )

    return an_quivers + dn_quivers + An_quivers
    #return an_quivers, dn_quivers


class HasseDiagram:
    def __init__(self, quiver = None):
        self.graph = self.build_graph(quiver)


    def build_graph(self, starting_quiver):
        #an_quivers, dn_quivers = init_minimal_transitions()
        minimal_transitions = init_minimal_transitions()

        adjacency_dict = {}
        curr_quivers = [starting_quiver]
        for _ in range(7):

            new_quivers = []

            for curr_quiver in curr_quivers:
                adjacency_dict[curr_quiver] = []

                for minimal_transition in minimal_transitions:

                    new = curr_quiver.subtract(minimal_transition)

                    if new: 
                        new_quivers.extend(new)
                        adjacency_dict[curr_quiver].extend(new)

            curr_quivers = new_quivers
        
        graph = nx.Graph()

        quiver_to_index = {}
        next_idx = 0
        for parent, children in adjacency_dict.items():
            if parent not in quiver_to_index:
                quiver_to_index[parent] = next_idx
                next_idx += 1

            for child in children:
                if child not in quiver_to_index:
                    quiver_to_index[child] = next_idx
                    next_idx += 1

        # DONT DO THIS AFTERWARDS BUT DURING SUBTRACITON
        quivers_by_dimension = {}
        for quiver, index in quiver_to_index.items():

            dimension = quiver.get_coulomb_dimension()
            if dimension not in quivers_by_dimension:
                quivers_by_dimension[dimension] = [quiver]
            else:
                quivers_by_dimension[dimension].append(quiver)


        # Now build the mapping duplicate_index -> rep_index
        removed_to_kept = {}
        for dim, quivers in quivers_by_dimension.items():

            reps = []  # will hold one canonical quiver per class
            for q in quivers:
                # see if q matches any existing rep
                for r in reps:
                    if q.is_equivalent(r):
                        # record: this q should be replaced by r
                        removed_to_kept[ quiver_to_index[q] ] = quiver_to_index[r]
                        break
                else:
                    # no match ⇒ q is a new representative
                    reps.append(q) 


        graph = nx.Graph()
        for quiver, index in quiver_to_index.items():

            if index in removed_to_kept:
                actual_index = removed_to_kept[index]
            else:
                actual_index = index

            if actual_index not in graph.nodes():
                graph.add_node(actual_index)
                graph.nodes[actual_index]['dimension'] = quiver.get_coulomb_dimension()

        for parent, children in adjacency_dict.items():
            u = quiver_to_index[parent]
            u = removed_to_kept[u] if u in removed_to_kept else u

            for child in children:
                v = quiver_to_index[child]
                v = removed_to_kept[v] if v in removed_to_kept else v

                graph.add_edge(u, v)

        return graph

                    















        


