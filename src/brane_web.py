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

    def subweb_decompositions(self) -> list:
        '''
        Finds all possible subweb decompositions of the brane web.

        A subweb decomposition is defined as a set of subwebs that together
        reconstruct the original brane web.
        '''

        subwebs = self.subwebs()
        queue = [(self.web, [])] # format: (graph: nx.MultiGraph, decomposition: List[List of edges])

        decompositions = []
        while queue:
            curr_graph, curr_decomp = queue.pop(0)

            subtraction_completed = False
            for subweb in subwebs:
                new_graph = curr_graph.copy()
                new_decomp = curr_decomp.copy()

                edges_count = collections.Counter(new_graph.edges())
                required_edges_count = collections.Counter(subweb)

                if not all(edges_count[edge] >= required_edges_count[edge] for edge in required_edges_count): continue

                new_graph = self.remove_edges(new_graph, subweb)
                new_decomp.append(subweb)
                subtraction_completed = True


                for g, d in queue:
                    if nx.is_isomorphic(new_graph, g) and all(any(collections.Counter(s) == collections.Counter(t) for t in d) for s in new_decomp):
                        break
                else:
                    queue.append((new_graph, new_decomp))

            if not subtraction_completed and curr_graph.number_of_edges() == 0:
                decompositions.append(curr_decomp)

        # minimal decompositions only
        all_subwebs = [] # [(subweb_edges, decomposition_idx),...]
        for i, decomp in enumerate(decompositions):
            for subweb in decomp:
                all_subwebs.append((subweb, i))

        to_remove = set()
        for i, (subweb1, idx1) in enumerate(all_subwebs):
            for j, (subweb2, idx2) in enumerate(all_subwebs):
                if i == j or idx1 == idx2: continue # skip if same subweb or same decomposition

                subweb1_issubset = all(item in subweb2 for item in subweb1)
                if subweb1_issubset and len(subweb1) < len(subweb2):
                    to_remove.add(idx2)

        minimal_decompositions = [decomp for i, decomp in enumerate(decompositions) if i not in to_remove]

        return minimal_decompositions

    def remove_edges(self, graph: nx.MultiGraph, edges: list) -> nx.MultiGraph:
        '''Removes edges from a graph, taking into account multiplicities.'''
        new_graph = graph.copy()

        edges_count = collections.Counter(new_graph.edges())
        edges_to_remove_count = collections.Counter(edges)

        for edge, count in edges_to_remove_count.items():
            if edges_count[edge] < count:
                raise ValueError(f"Cannot remove {count} instances of edge {edge}, only {edges_count[edge]} exist.")

            # keys
            keys = list(new_graph[edge[0]][edge[1]].keys())
            for key in keys[:count]:
                new_graph.remove_edge(edge[0], edge[1], key=key)

        return new_graph

    def subwebs(self) -> list:
        '''Finds all possible and valid subwebs of the brane web.'''

        branes = list(self.web.edges())

        candidates = []
        for r in range(1, len(branes)+1):
            combinations_of_size_r = itertools.combinations(branes, r)

            for combination in combinations_of_size_r:
                current_graph = nx.MultiGraph(combination)

                if not nx.is_connected(current_graph): continue
                if not self.conserves_charge(combination): continue
                if combination in candidates: continue
                if self.violates_srule(combination, current_graph): continue

                candidates.append(combination)
            
        return candidates 

    def violates_srule(self, branes: list, subweb: nx.MultiGraph) -> bool:
        '''Checks if a set of branes violates the S-rule.'''

        # extracting junctions from branes
        junctions = set()
        for u, v in branes:
            if self.web.nodes[u]['type'] == 'junction':
                junctions.add(u)
            if self.web.nodes[v]['type'] == 'junction':
                junctions.add(v)

        # computing total NS5 charge of subweb, this assumes no 7-branes in between junctions!!!
        NS5_charge = 0
        for junction in junctions:
            branes_at_junction = [brane for brane in branes if junction in brane]
            NS5_charge_junction = 0

            for brane in branes_at_junction:
                u, v = brane
                other_node = v if u == junction else u
                charge_into_junction = self.charge_into_node(junction, other_node)

                NS5_charge_junction += abs(charge_into_junction[1])

            NS5_charge_junction = NS5_charge_junction // 2
            NS5_charge += NS5_charge_junction

        # checking if s-rule is violated
        branes_between = [] # all branes between 7-branes and junctions
        for u, v in branes:
            if self.web.nodes[u]['type'] == 'seven-brane' and self.web.nodes[v]['type'] == 'junction':
                branes_between.append((u, v))
            if self.web.nodes[v]['type'] == 'seven-brane' and self.web.nodes[u]['type'] == 'junction':
                branes_between.append((v, u))

        brane_counts = collections.Counter(branes_between)
        for (u, v), count in brane_counts.items():
            excess = count - NS5_charge
            if excess <= 0: continue

            if not self.extension_exists(subweb ,(u, v), excess, NS5_charge):
                return True

        return False

    def extension_exists(self, subweb: nx.MultiGraph, brane: tuple, excess: int, NS5_charge: int) -> bool:
        '''
        Checks if branes extend over 7-branes to satisfy the S-rule.

        brane: (7-brane to extend over, junction/7-brane) IMPORTANT!
        '''

        seven_brane, other = brane

        possible_extensions = list(subweb.edges(seven_brane))
        possible_extensions = [edge for edge in possible_extensions if edge[1] != other]
        possible_extensions_counts = collections.Counter(possible_extensions)

        extension = None
        for (u, v), count in possible_extensions_counts.items():
            if count >= excess:
                extension = (v, u)
                new_excess = excess - NS5_charge
                break
        else:
            return False

        if new_excess <= 0:
            return True 
        
        return self.extension_exists(subweb, extension, new_excess, NS5_charge)

    def conserves_charge(self, branes: list) -> bool:
        '''Checks if a set of branes conserves charge at each junction.'''

        # extracting junctions from branes
        junctions = set()
        for u, v in branes:
            if self.web.nodes[u]['type'] == 'junction':
                junctions.add(u)
            if self.web.nodes[v]['type'] == 'junction':
                junctions.add(v)

        for junction in junctions:
            branes_at_junction = [brane for brane in branes if junction in brane]
            total_charge = [0, 0]

            for brane in branes_at_junction:
                u, v = brane
                other_node = v if u == junction else u
                charge_into_junction = self.charge_into_node(junction, other_node)

                total_charge[0] += charge_into_junction[0]
                total_charge[1] += charge_into_junction[1]

            if total_charge != [0, 0]:
                return False
            
        return True

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

    @classmethod
    def from_subgraph_edges(cls, graph: nx.MultiGraph, edges: list):
        '''Creates a BraneWeb instance from a subgraph defined by a list of edges in the original graph.'''

        edges_counter = collections.Counter(edges)

        upd_edges = []
        for (u, v), count in edges_counter.items():
            for i in range(count):
                upd_edges.append((u, v, i))
        
        subgraph = graph.edge_subgraph(upd_edges).copy()
        return cls.from_graph(subgraph)


