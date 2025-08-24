import math
import itertools
import collections

import networkx as nx
import matplotlib.pyplot as plt


class BraneWeb:
    '''
    A class to represent a brane web using a graph structure.
    '''

    def __init__(self):
        self.web = nx.MultiGraph()

    def add_junction(self, junction_id: str, position: tuple):
        '''Adds a junction to the brane web (a type of node).'''
        self.web.add_node(junction_id, pos=position, type='junction')

    def add_seven_brane(self, brane_id: str, position: tuple):
        '''Adds a 7-brane to the brane web (a type of node).'''
        self.web.add_node(brane_id, pos=position, type='seven-brane')

    def add_brane(self, start_id: str, end_id: str, multiplicity: int = 1):
        '''Adds a brane (an edge) between two nodes in the brane web.'''
        if start_id not in self.web.nodes or end_id not in self.web.nodes:
            raise ValueError("Both start and end nodes must exist in the brane web.")

        charge = self.charge_into_node(end_id, start_id)
        for _ in range(multiplicity):
            self.web.add_edge(start_id, end_id, charge=charge)

    def find_subwebs_across_junction(self, junction: str) -> list:
        '''
        Finds all possible subwebs that can be formed across a junction.

        A subweb across a junction is defined as a set of branes connected to the junction
        that conserves charge.
        '''

        branes_at_junction = list(self.web.edges(junction)) # in the formate (junction, x)

        candidates = []
        for r in range(1, len(branes_at_junction)+1):
            combs_of_size_r = itertools.combinations(branes_at_junction, r)

            for comb in combs_of_size_r:
                if not self.conserves_charge(comb):
                    continue

                candidates.append(comb)
        candidates_sorted_inner = [tuple(sorted(candidate)) for candidate in candidates]
        candidates_sorted = sorted(candidates_sorted_inner)

        # removing duplicates
        unique_candidates = []
        for candidate in candidates_sorted:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)

        for candidate in unique_candidates:
            print('candidate = ', candidate)
        print('len valid candidates = ', len(unique_candidates))

        # minimal candiates only
        unique_candidates_set = sorted([set(candidate) for candidate in unique_candidates])
        minimal_candidates = unique_candidates_set.copy()

        for i, s in enumerate(unique_candidates_set):

            for j, t in enumerate(unique_candidates_set):
                if j <= i: continue

                if s.issubset(t):
                    print(f'removing candidate {t} as it is not minimal, since it contains {s}')
                    if t in minimal_candidates:
                        minimal_candidates.remove(t)

        print('len minimal candidates = ', len(minimal_candidates))

        subwebs = []
        for candidate in minimal_candidates:
            candidate = list(candidate)
            brane_counts = collections.Counter(candidate)

            upd_candidate = []
            for (u, v), count in brane_counts.items():
                for i in range(count):
                    upd_candidate.append(
                        (u, v, i)
                    )

            subweb_graph = self.web.edge_subgraph(upd_candidate).copy()
            subweb = BraneWeb.from_graph(subweb_graph)

            subwebs.append(subweb)

        return subwebs

    def conserves_charge(self, branes: list) -> bool:
        '''Checks if a set of branes conserves charge.'''
        total_charge = [0, 0]

        for brane in branes:
            u, x = brane
            charge_into_u = self.charge_into_node(u, x)

            total_charge[0] += charge_into_u[0]
            total_charge[1] += charge_into_u[1]

        return total_charge == [0, 0]

    def charge_into_node(self, node: str, other_node: str) -> tuple:
        '''Calculates the charge vector pointing into a node from another node.'''
        if node not in self.web.nodes or other_node not in self.web.nodes:
            raise ValueError("Both nodes must exist in the brane web.")

        charge = (
            self.web.nodes[node]['pos'][0] - self.web.nodes[other_node]['pos'][0],
            self.web.nodes[node]['pos'][1] - self.web.nodes[other_node]['pos'][1]
        )
        gcd = math.gcd(int(charge[0]), int(charge[1]))
        charge = (charge[0] // gcd, charge[1] // gcd)

        return charge

    def draw(self, save_path: str = None):
        '''Visualises the brane web.'''
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

        pos = nx.get_node_attributes(self.web, 'pos')

        junctions = [n for n, attr in self.web.nodes(data=True) if attr['type'] == 'junction']
        seven_branes = [n for n, attr in self.web.nodes(data=True) if attr['type'] == 'seven-brane']

        nx.draw_networkx_edges(self.web, pos, ax=ax, width=3, edge_color='black')

        nx.draw_networkx_nodes(self.web, pos, nodelist=junctions, node_color='black', node_size=30, label='Junctions', ax=ax)
        nx.draw_networkx_nodes(self.web, pos, nodelist=seven_branes, node_color='gray', node_size=300, label='7-Branes', ax=ax)

        # drawing charges on branes
        for edge in self.web.edges(data=True):
            u, v, data = edge

            vec = (pos[v][0] - pos[u][0], pos[v][1] - pos[u][1])
            length = (vec[0]**2 + vec[1]**2)**0.5
            angle = math.degrees(math.atan2(vec[1], vec[0]))
            if 90 <= angle <= 180:
                angle -= 180
            elif -180 < angle < -90:
                angle += 180

            if length == 0:
                continue

            norm_vec = (vec[0] / length, vec[1] / length)
            perp_vec = (-norm_vec[1], norm_vec[0])
            offset = 0.07

            position = (
                pos[u][0] + 0.5*vec[0] + offset*perp_vec[0],
                pos[u][1] + 0.5*vec[1] + offset*perp_vec[1]
            )

            ax.text(position[0], position[1], f"{data['charge']}", rotation=angle, color='black', fontsize=10, ha='center', va='center')

        # drawing multiplicities on branes
        edges = list(self.web.edges())

        edges_sorted_inner = [tuple(sorted(edge)) for edge in edges]
        edges_sorted = sorted(edges_sorted_inner)
        edges_counts = collections.Counter(edges_sorted)

        for (u, v), count in edges_counts.items():
            if count == 1:
                continue

            vec = (pos[v][0] - pos[u][0], pos[v][1] - pos[u][1])
            length = (vec[0]**2 + vec[1]**2)**0.5
            angle = math.degrees(math.atan2(vec[1], vec[0]))
            if 90 <= angle <= 180:
                angle -= 180
            elif -180 < angle < -90:
                angle += 180

            if length == 0:
                continue

            norm_vec = (vec[0] / length, vec[1] / length)
            perp_vec = (-norm_vec[1], norm_vec[0])
            offset = -0.07

            position = (
                pos[u][0] + 0.5*vec[0] + offset*perp_vec[0],
                pos[u][1] + 0.5*vec[1] + offset*perp_vec[1]
            )

            ax.text(position[0], position[1], f"x{count}", rotation=angle, color='black', fontsize=10, ha='center', va='center')

        ax.set_aspect('equal')
        ax.grid(False)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()

    @classmethod
    def from_graph(cls, graph: nx.MultiGraph):
        '''Creates a BraneWeb instance from an existing NetworkX MultiGraph.'''
        brane_web = cls()
        brane_web.web = graph
        return brane_web




