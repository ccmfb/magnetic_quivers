import networkx as nx
import matplotlib.pyplot as plt


class BraneWeb:
    '''
    A class to represent a brane web using a graph structure.
    '''

    def __init__(self):
        self.web = nx.Graph()

    def add_junction(self, junction_id: str, position: tuple):
        '''Adds a junction to the brane web (a type of node).'''
        self.web.add_node(junction_id, pos=position, type='junction')

    def add_seven_brane(self, brane_id: str, position: tuple):
        '''Adds a 7-brane to the brane web (a type of node).'''
        self.web.add_node(brane_id, pos=position, type='seven-brane')

    def add_brane(self, start_id: str, end_id: str, charge: tuple, multiplicity: int = 1):
        '''Adds a brane (an edge) between two nodes in the brane web.'''
        if start_id not in self.web.nodes or end_id not in self.web.nodes:
            raise ValueError("Both start and end nodes must exist in the brane web.")

        self.web.add_edge(start_id, end_id, charge=charge, multiplicity=multiplicity)

    def extract_subwebs(self):
        '''Extracts and returns all connected sub-webs from the brane web.'''
        subweb_nodes = list(nx.connected_components(self.web))
        print(subweb_nodes)
        extracted_subwebs = []

        for nodes in subweb_nodes:
            subgraph = self.web.subgraph(nodes).copy()

            subweb = BraneWeb()
            subweb.web = subgraph
            extracted_subwebs.append(subweb)

        return extracted_subwebs

    def draw(self, save_path: str = None):
        '''Visualises the brane web.'''
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

        pos = nx.get_node_attributes(self.web, 'pos')

        junctions = [n for n, attr in self.web.nodes(data=True) if attr['type'] == 'junction']
        seven_branes = [n for n, attr in self.web.nodes(data=True) if attr['type'] == 'seven-brane']

        nx.draw_networkx_edges(self.web, pos, ax=ax, width=3, edge_color='black')

        nx.draw_networkx_nodes(self.web, pos, nodelist=junctions, node_color='black', node_size=30, label='Junctions', ax=ax)
        nx.draw_networkx_nodes(self.web, pos, nodelist=seven_branes, node_color='gray', node_size=300, label='7-Branes', ax=ax)

        for edge in self.web.edges(data=True):
            u, v, data = edge

            vec = (pos[v][0] - pos[u][0], pos[v][1] - pos[u][1])
            length = (vec[0]**2 + vec[1]**2)**0

            if length == 0:
                continue

            norm_vec = (vec[0] / length, vec[1] / length)
            perp_vec = (-norm_vec[1], norm_vec[0])
            offset = 0.1

            position = (
                pos[u][0] + 0.5*vec[0] + offset*perp_vec[0],
                pos[u][1] + 0.5*vec[1] + offset*perp_vec[1]
            )

            ax.text(position[0], position[1], f"{data['multiplicity']}{data['charge']}", color='black', fontsize=10, ha='center', va='center')

        ax.grid(False)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()





