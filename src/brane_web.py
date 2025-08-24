import math
import itertools

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

    def add_brane(self, start_id: str, end_id: str, multiplicity: int = 1):
        '''Adds a brane (an edge) between two nodes in the brane web.'''
        if start_id not in self.web.nodes or end_id not in self.web.nodes:
            raise ValueError("Both start and end nodes must exist in the brane web.")

        charge = (
            self.web.nodes[end_id]['pos'][0] - self.web.nodes[start_id]['pos'][0],
            self.web.nodes[end_id]['pos'][1] - self.web.nodes[start_id]['pos'][1]
        )
        gcd = math.gcd(int(charge[0]), int(charge[1]))
        charge = (charge[0] // gcd, charge[1] // gcd)

        self.web.add_edge(start_id, end_id, charge=charge, multiplicity=multiplicity)

    def find_subweb_decompositions(self, start_brane) -> list:
        '''
        Finds all possible subweb decompositions of the brane web.

        as of now needs an edge to start from... probably can be improved by starting from junction..
        '''

        # find all edges connected to the start edge
        u, v = start_brane
        branes_at_u = list(self.web.edges(u))
        if (u, v) in branes_at_u: branes_at_u.remove((u, v))
        if (v, u) in branes_at_u: branes_at_u.remove((v, u))
        start_charge_into_u = self.charge_into_node(u, v)

        branes_at_v = list(self.web.edges(v)) # in the formate (v, x)
        if (u, v) in branes_at_v: branes_at_v.remove((u, v))
        if (v, u) in branes_at_v: branes_at_v.remove((v, u))
        start_charge_into_v = self.charge_into_node(v, u)

        candidates = []

        for r in range(1, len(branes_at_u)+1):
            combs_of_size_r = itertools.combinations(branes_at_u, r)

            for comb in combs_of_size_r:
                # check if the charges sum to zero
                total_charge = [0, 0]
                total_charge[0] += start_charge_into_u[0]
                total_charge[1] += start_charge_into_u[1]

                for brane in comb:
                    charge_into_u = self.charge_into_node(u, brane[1])

                    total_charge[0] += charge_into_u[0]
                    total_charge[1] += charge_into_u[1]

                if total_charge != [0, 0]:
                    continue

                valid_comb = [(u, v)] + list(comb)
                candidates.append(valid_comb)

        for r in range(1, len(branes_at_v)+1):
            combs_of_size_r = itertools.combinations(branes_at_v, r)

            for comb in combs_of_size_r:
                # check if the charges sum to zero
                total_charge = [0, 0]
                total_charge[0] += start_charge_into_v[0]
                total_charge[1] += start_charge_into_v[1]

                for brane in comb:
                    charge_into_v = self.charge_into_node(v, brane[1])

                    total_charge[0] += charge_into_v[0]
                    total_charge[1] += charge_into_v[1]

                if total_charge != [0, 0]:
                    continue

                valid_comb = [(u, v)] + list(comb)
                candidates.append(valid_comb)
                    
        print()
        print('valied candidates = ', candidates)



        return []

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

        for edge in self.web.edges(data=True):
            u, v, data = edge

            vec = (pos[v][0] - pos[u][0], pos[v][1] - pos[u][1])
            length = (vec[0]**2 + vec[1]**2)**0
            angle = math.degrees(math.atan2(vec[1], vec[0]))
            if 90 <= angle <= 180:
                angle -= 180
            elif -180 < angle < -90:
                angle += 180

            if length == 0:
                continue

            norm_vec = (vec[0] / length, vec[1] / length)
            perp_vec = (-norm_vec[1], norm_vec[0])
            offset = 0.1

            position = (
                pos[u][0] + 0.5*vec[0] + offset*perp_vec[0],
                pos[u][1] + 0.5*vec[1] + offset*perp_vec[1]
            )

            ax.text(position[0], position[1], f"{data['multiplicity']}{data['charge']}", rotation=angle, color='black', fontsize=10, ha='center', va='center')

        ax.set_aspect('equal')
        ax.grid(False)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()





