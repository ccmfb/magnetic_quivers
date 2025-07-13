import networkx as nx
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt


class Quiver:
    def __init__(self, edges: list, node_values: list, name: str = None):
        '''Initialize a Quiver with edges and node values.
        
        Args:
            edges (list): List of tuples representing edges between nodes.
            node_values (list): List of values for each node. In order of node IDs.
        '''

        self.quiver = nx.MultiGraph()
        self.quiver.add_edges_from(edges)

        nodes = list(self.quiver.nodes())
        nodes.sort()
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


    def subtract(self, sub_quiver):
        '''
        Subtract the values of the nodes in 'sub_quiver' from this quiver. If sub_quiver is not found in this quiver, return the original quiver.

        Args:
            sub_quiver (Quiver): The subgraph to subtract.

        Returns:
            Quiver: A new Quiver with the values subtracted.
        '''

        mappings = self.find_all_embeddings(sub_quiver)
        if not mappings: return [self]  # No embeddings found, return original quiver

        results = []
        H = sub_quiver.quiver
        for mapping in mappings:
            G = self.quiver.copy() # New quiver for each mapping
            old_balance = self.get_balance()

            for h_node, g_node in mapping.items():
                G.nodes[g_node]['value'] -= H.nodes[h_node]['value']

                if G.nodes[g_node]['value'] <= 0:
                    G.remove_node(g_node)

            new_quiver = Quiver.from_graph(G, name=f"{self.name} - {sub_quiver.name}")
            new_quiver = new_quiver.restore_balance(old_balance)
            results.append(new_quiver)

        if results:
            return results
        else:
            return [self]


    def restore_balance(self, old_balance):
        '''
        Adds an overall U(1) node to the quiver and adds edges to restore the balance of nodes back to their pre-subtraction
        values.
        '''
        graph = self.quiver.copy()
        curr_balance = self.get_balance()

        additional_node = max(graph.nodes()) + 1
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
        edges = list(graph.edges(keys=True))
        node_values = [graph.nodes[node]['value'] for node in graph.nodes()]
        return Quiver(edges, node_values, name=name)


def init_minimal_transitions(dim_upper_bound: int = 10):
    '''Some minimal nilpotent orbit transitions...'''

    an_quivers = []
    an_upper_bound = dim_upper_bound
    for n in range(1, an_upper_bound + 1):
        linear_edges = [(i, i+1) for i in range(1, n)]
        all_edges = linear_edges + [(n, n+1)] + [(n+1, 1)]
        values = [1] * (n + 1)
        # print(f"Creating a_{n} quiver with edges: {all_edges}")

        an_quivers.append(
            Quiver(all_edges, values, name=f"a_{n}")
        )

    dn_quivers = []
    #dn_upper_bound = (dim_upper_bound + 3) / 2 if (dim_upper_bound + 3) % 2 == 0 else (dim_upper_bound + 2) / 2
    dn_upper_bound = 5
    for n in range(4, dn_upper_bound + 1):
        linear_edges = [(i, i+1) for i in range(1, n-1)]
        all_edges = linear_edges + [(2, n)] + [(n-2, n+1)]
        values = [1] + [2] * (n - 3) + [1] + [1, 1]

        dn_quivers.append(
            Quiver(all_edges, values, name=f"d_{n}")
        )

    return an_quivers, dn_quivers


