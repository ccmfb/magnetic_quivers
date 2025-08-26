import math
import itertools
import collections

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiLineString
from shapely.strtree import STRtree

from unitary_quiver import Quiver


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

    def find_magnetic_quivers(self):
        '''
        Finds the magnetic quivers associated with the brane web. 
        '''

        subweb_decompositions = self.find_subweb_decompositions('j1', draw_subwebs=False)

        magnetic_quivers = []
        for i,decomp in enumerate(subweb_decompositions):

            subweb_counts = {}
            for j, subweb in enumerate(decomp):
                edges = tuple(subweb.edges())

                if edges not in subweb_counts:
                    subweb_counts[edges] = 0
                subweb_counts[edges] += 1

            number_of_nodes = len(subweb_counts)

            nodes = [i+1 for i in range(number_of_nodes)]
            values = [count for count in subweb_counts.values()]
            edges = []

            for i, subweb in enumerate(subweb_counts.keys()):
                for j, other_subweb in enumerate(subweb_counts.keys()):
                    if i <= j: continue

                    multiplicity1 = subweb_counts[subweb]
                    multiplicity2 = subweb_counts[other_subweb]

                    subweb_with_keys = [(u, v, 0) for u, v in subweb]
                    other_subweb_with_keys = [(u, v, 0) for u, v in other_subweb]
                    subgraph1 = self.web.edge_subgraph(subweb_with_keys)
                    subgraph2 = self.web.edge_subgraph(other_subweb_with_keys)

                    edge_num = self.edge_number(subgraph1, subgraph2, multiplicity1, multiplicity2)
                    edge_num = edge_num // (multiplicity1 * multiplicity2)

                    for _ in range(edge_num):
                        edges.append((i+1, j+1))

            magnetic_quivers.append(
                Quiver(nodes, edges, values)
            )
        
        return magnetic_quivers

    def edge_number(self, subweb1, subweb2, multiplicity1, multiplicity2) -> int:
        '''
        Calculates the number of edges between two subwebs.
        '''

        # Graphs to shapely MultiLineStrings, one slightly offset for stable intersection
        geometry1 = self.graph_to_multilinestring(subweb1)
        geometry2 = self.graph_to_multilinestring(subweb2, offset=True)


        # Finding intersections using STRtree for efficiency
        all_lines1 = list(geometry1.geoms)
        all_lines2 = list(geometry2.geoms)
        tree2 = STRtree(all_lines2)

        intersection_pairs = []
        for i, line1 in enumerate(all_lines1):
            candidates = tree2.query(line1)

            for j in candidates:
                candidate_line2 = all_lines2[j]

                if line1.intersects(candidate_line2):
                    intersection_pairs.append(
                        (line1, candidate_line2)
                    )

        # Calculating intersection number
        intersection_number = 0
        for line1, line2 in intersection_pairs:
            line1 = list(line1.coords)
            line2 = list(line2.coords)

            charges1 = (
                int(line1[1][0] - line1[0][0]),
                int(line1[1][1] - line1[0][1])
            )
            charges1 = (charges1[0] * multiplicity1, charges1[1] * multiplicity1)
            charges2 = (
                int(line2[1][0] - line2[0][0]),
                int(line2[1][1] - line2[0][1])
            )
            charges2 = (charges2[0] * multiplicity2, charges2[1] * multiplicity2)

            determinant = charges1[0] * charges2[1] - charges1[1] * charges2[0]
            intersection_number += abs(determinant)

        # 7-brane corrections, needs implementation
        shared_nodes = set(subweb1.nodes()) & set(subweb2.nodes())
        for node in shared_nodes:
            if self.web.nodes[node]['type'] != 'seven-brane': continue

            edges1 = list(subweb1.edges(node))
            edges2 = list(subweb2.edges(node))

            charge_into_node1 = self.charge_into_node(node, edges1[0][1] if edges1[0][0] == node else edges1[0][0])
            charge_into_node1 = np.array(charge_into_node1)

            charge_into_node2 = self.charge_into_node(node, edges2[0][1] if edges2[0][0] == node else edges2[0][0])
            charge_into_node2 = np.array(charge_into_node2)

            if np.dot(charge_into_node1, charge_into_node2) > 0:
                # same direction, subtract
                intersection_number -= multiplicity1 * multiplicity2
            else:
                # opposite direction, add
                intersection_number += multiplicity1 * multiplicity2

        return intersection_number

    def graph_to_multilinestring(self, graph: nx.MultiGraph, offset=False) -> MultiLineString:
        '''Converts a NetworkX MultiGraph to a Shapely MultiLineString.'''

        if offset:
            offset_vec = np.random.rand(2)
            length = np.linalg.norm(offset_vec)
            offset_vec = 0.1 * (offset_vec / length)
        else:
            offset_vec = np.array([0, 0])

        lines = []
        for u, v in graph.edges():
            pos_u = np.array(self.web.nodes[u]['pos']) + offset_vec
            pos_v = np.array(self.web.nodes[v]['pos']) + offset_vec
            lines.append([(pos_u[0], pos_u[1]), (pos_v[0], pos_v[1])])

        return MultiLineString(lines)

    def find_subweb_decompositions(self, junction: str, draw_subwebs: bool = False) -> list:
        '''
        Finds all possible subweb decompositions of the brane web.

        A subweb decomposition is defined as a set of subwebs that together
        reconstruct the original brane web.
        '''

        minimal_subwebs = self.find_subwebs_across_junction(junction)

        queue = [(self.web, [])] # format: (graph: nx.MultiGraph, decomposition: List[nx.MultiGraph])
        subweb_decompositions_junction = []

        while queue:

            curr_graph, curr_decomp = queue.pop(0)

            subtraction_done = False
            for subweb in minimal_subwebs:
                web_copy = curr_graph.copy()
                decomp_copy = curr_decomp.copy()
                all_edges_exist = all(web_copy.has_edge(u, v, key=k) for u, v, k in subweb.web.edges(keys=True))

                # subtracting subweb from web_copy, if exists
                if not all_edges_exist: continue
                web_copy.remove_edges_from(subweb.web.edges())
                subtraction_done = True

                # check if result is already in queue
                for queue_graph, _ in queue:
                    if nx.is_isomorphic(web_copy, queue_graph):
                        break
                else:
                    decomp_copy.append(subweb.web)
                    queue.append((web_copy, decomp_copy))

            if not subtraction_done:
                subweb_decompositions_junction.append(
                    (curr_graph, curr_decomp)
                )

        # check for disconnected subwebs in the remaining graph
        subweb_decompositions = []
        for graph, decomp in subweb_decompositions_junction:

            # check it there are further edges
            if graph.number_of_edges() == 0:
                subweb_decompositions.append((graph, decomp))
                continue 

            curr_graph = graph.copy()
            curr_decomp = decomp.copy()

            for u, v, k in graph.edges(keys=True):
                edge_subgraph = curr_graph.edge_subgraph([(u, v, k)]).copy()
                curr_decomp.append(edge_subgraph)
                curr_graph.remove_edge(u, v, k)

            subweb_decompositions.append((curr_graph, curr_decomp))

        if draw_subwebs:
            for i, (graph, decomp) in enumerate(subweb_decompositions):
                print(f"Decomposition {i+1}:")
                graph_web = BraneWeb.from_graph(graph)
                graph_web.draw()

                for j, subweb in enumerate(decomp):
                    print(f" Subweb {j+1}:")
                    subweb_instance = BraneWeb.from_graph(subweb)
                    subweb_instance.draw()

        subweb_decompositions = [decomp for _, decomp in subweb_decompositions]

        return subweb_decompositions

    def srule_adjusted_decompositions(self, decompositions: list) -> list:
        '''
        Adjusts decompositions according to the S-rule.
        
        The S-rule states that no two branes can end on the same 7-brane and NS5-brane.
        '''

        # Check if s-rule is violated in any decomposition
        for decomp in decompositions:
            for subweb in decomp:
                print("Subweb:", subweb.edges())

                NS5_charge = 0
                D5_charge = 0 # not actual D5 charge, just for info
                for edge in subweb.edges(data=True):
                    NS5_charge += abs(edge[2]['charge'][1])
                    D5_charge += abs(edge[2]['charge'][0])
                NS5_charge = NS5_charge // 2 
                print("  NS5 charge:", NS5_charge)

                nodes_from_edges = []
                for u, v in subweb.edges():
                    nodes_from_edges.append(u)
                    nodes_from_edges.append(v)

                nodes_counts = collections.Counter(nodes_from_edges)
                print("  Nodes:", nodes_counts)

                for node, count in nodes_counts.items():
                    if self.web.nodes[node]['type'] != 'seven-brane': continue

                    if count > NS5_charge and D5_charge > 0 and NS5_charge > 0:
                        print("  S-rule violated at node", node, "with count", count, "and NS5 charge", NS5_charge)

            print("-----")

        return decompositions # placeholder, needs implementation

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

                if not self.conserves_srule(comb):
                    continue

                candidates.append(comb)
        candidates_sorted_inner = [tuple(sorted(candidate)) for candidate in candidates]
        candidates_sorted = sorted(candidates_sorted_inner)

        # removing duplicates
        unique_candidates = []
        for candidate in candidates_sorted:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)

        # minimal candiates only
        unique_candidates_set = sorted([sorted(candidate) for candidate in unique_candidates])
        minimal_candidates = unique_candidates_set.copy()

        for i, s in enumerate(unique_candidates_set):
            for j, t in enumerate(unique_candidates_set):
                if j == i: continue

                s_issubset = all(item in t for item in s)
                if s_issubset and s != t and len(s) < len(t):
                    if t in minimal_candidates:
                        minimal_candidates.remove(t)

        # constructing subwebs from data, might not be neccessary
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

        return subwebs # fix: return graphs instead of web..

    def conserves_srule(self, branes: list) -> bool:
        '''Checks if a set of branes conserves the S-rule.'''

        # Checking for duplicates branes
        brane_counts = collections.Counter(branes)

        for brane, count in brane_counts.items():
            if count > 1:
                print('-'*20)
                print("Subweb:", branes)
                print("  S-rule violated due to duplicate brane:", brane)

        return True # placeholder, needs implementation

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




