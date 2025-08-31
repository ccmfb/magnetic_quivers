import math
import itertools
import collections
from dataclasses import dataclass

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
    Brane = tuple[str, str]
    Subweb = list[Brane]

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

    def magnetic_quivers(self) -> list[Quiver]:
        '''Finds the magnetic quivers associated with a brane web.'''

        decompositions = self.decompositions()

        magnetic_quivers = []
        for decomposition in decompositions:
            decomposition = [tuple(subweb) for subweb in decomposition]
            subweb_counts = collections.Counter(decomposition)

            number_of_nodes = len(subweb_counts)
            nodes = [i+1 for i in range(number_of_nodes)]
            values = [count for count in subweb_counts.values()]
            edges = []

            for i, subweb1 in enumerate(subweb_counts.keys()):
                for j, subweb2 in enumerate(subweb_counts.keys()):
                    if i <= j: continue

                    count1 = subweb_counts[subweb1]
                    count2 = subweb_counts[subweb2]

                    edge_number = self.edge_number(subweb1, subweb2, count1, count2)

                    if edge_number % (count1 * count2) != 0:
                        raise ValueError("Edge number not divisible by product of counts, something went wrong.")

                    edge_number = edge_number // (count1 * count2)

                    for _ in range(edge_number):
                        edges.append((i+1, j+1))

            magnetic_quivers.append(
                Quiver(nodes, edges, values)
            )
        
        return magnetic_quivers

    def edge_number(self, subweb1: Subweb, subweb2: Subweb, count1: int, count2: int) -> int:
        '''Calculates the number of edges between two nodes, ie. subwebs'''

        geometry1 = self.subweb_to_multilinestring(subweb1)
        geometry2 = self.subweb_to_multilinestring(subweb2, offset=True)

        # Finding intersections
        all_lines1 = list(geometry1.geoms)
        all_lines2 = list(geometry2.geoms)
        tree2 = STRtree(all_lines2)

        intersection_pairs = []
        for line1 in all_lines1:
            candidates = tree2.query(line1)

            for index in candidates:
                candidate_line2 = all_lines2[index]

                if not line1.intersects(candidate_line2): continue

                intersection_pairs.append((line1, candidate_line2))

        # Calculating intersection number
        intersection_number = 0
        for line1, line2 in intersection_pairs:
            line1 = list(line1.coords)
            line2 = list(line2.coords)

            charges1 = (
                int(line1[1][0] - line1[0][0]),
                int(line1[1][1] - line1[0][1])
            )
            charges1 = (charges1[0] * count1, charges1[1] * count1)
            charges2 = (
                int(line2[1][0] - line2[0][0]),
                int(line2[1][1] - line2[0][1])
            )
            charges2 = (charges2[0] * count2, charges2[1] * count2)

            determinant = charges1[0] * charges2[1] - charges1[1] * charges2[0]
            intersection_number += abs(determinant)

        # 7-brane corrections, possibly the ugliest code i have ever written
        seven_branes1 = set()
        for u, v in subweb1:
            if self.web.nodes[u]['type'] == 'seven-brane':
                seven_branes1.add(u)
            if self.web.nodes[v]['type'] == 'seven-brane':
                seven_branes1.add(v)

        seven_branes2 = set()
        for u, v in subweb2:
            if self.web.nodes[u]['type'] == 'seven-brane':
                seven_branes2.add(u)
            if self.web.nodes[v]['type'] == 'seven-brane':
                seven_branes2.add(v)

        shared_seven_branes = seven_branes1 & seven_branes2

        for seven_brane in shared_seven_branes:
            branes1_on_seven_brane_side1 = 0
            branes1_on_seven_brane_side2 = 0

            branes2_on_seven_brane_side1 = 0
            branes2_on_seven_brane_side2 = 0

            side1_node = None
            side2_node = None

            for u, v in subweb1:
                if u == seven_brane:
                    other_node = v
                elif v == seven_brane:
                    other_node = u
                else:
                    continue

                if side1_node == other_node: continue
                if side2_node == other_node: continue

                if side1_node == None:
                    side1_node = other_node
                elif side2_node == None:
                    side2_node = other_node

            for u, v in subweb2:
                if u == seven_brane:
                    other_node = v
                elif v == seven_brane:
                    other_node = u
                else:
                    continue

                if side1_node == other_node: continue
                if side2_node == other_node: continue

                if side1_node == None:
                    side1_node = other_node
                elif side2_node == None:
                    side2_node = other_node

            for u, v in subweb1:
                if u == seven_brane:
                    other_node = v
                elif v == seven_brane:
                    other_node = u
                else:
                    continue

                if other_node == side1_node:
                    branes1_on_seven_brane_side1 += count1
                if other_node == side2_node:
                    branes1_on_seven_brane_side2 += count1

            for u, v in subweb2:
                if u == seven_brane:
                    other_node = v
                elif v == seven_brane:
                    other_node = u
                else:
                    continue

                if other_node == side1_node:
                    branes2_on_seven_brane_side1 += count2
                if other_node == side2_node:
                    branes2_on_seven_brane_side2 += count2


            intersection_number += branes1_on_seven_brane_side1 * branes2_on_seven_brane_side2 # opposite sides
            intersection_number += branes2_on_seven_brane_side1 * branes1_on_seven_brane_side2 # opposite sides
            intersection_number -= branes1_on_seven_brane_side1 * branes2_on_seven_brane_side1 # same side
            intersection_number -= branes1_on_seven_brane_side2 * branes2_on_seven_brane_side2 # same side

        return intersection_number

    def subweb_to_multilinestring(self, subweb: Subweb, offset: bool = False) -> MultiLineString:
        '''Converts Subwebs into multilinestrings'''

        if offset:
            offset_vec = np.random.rand(2)
            length = np.linalg.norm(offset_vec)
            offset_vec = 0.1 * (offset_vec / length)
        else:
            offset_vec = np.array([0, 0])

        lines = []
        for u, v in subweb:
            position_u = np.array(self.web.nodes[u]['pos']) + offset_vec
            position_v = np.array(self.web.nodes[v]['pos']) + offset_vec
            lines.append(
                [position_u, position_v]
            )
        
        return MultiLineString(lines)

    def decompositions(self) -> list[list[Subweb]]:
        '''Finds all maximal subweb decompositions of the brane web.'''

        all_subwebs = self.subwebs()
        decompositions = []

        queue = [(self.web.edges(), [])] # fomat: (remaining_branes, list of subwebs found so far)

        while queue:
            remaining_branes, found_subwebs = queue.pop(0)
            if len(remaining_branes) == 0:
                for other_decomposition in decompositions:
                    if self.same_decomposition(found_subwebs, other_decomposition): break
                else:
                    decompositions.append(found_subwebs)

            for subweb in all_subwebs:
                if not self.is_subweb(subweb, remaining_branes): continue

                new_remaining_branes = self.subtract_subweb(remaining_branes, subweb)
                new_decomposition = found_subwebs.copy()
                new_decomposition.append(subweb)

                for _, other_decomposition in queue:
                    if self.same_decomposition(new_decomposition, other_decomposition): break
                else:
                    queue.append((new_remaining_branes, new_decomposition))

        return decompositions

    def subwebs(self) -> list[Subweb]:
        '''Finds all subwebs in the brane web.'''
        subwebs = []

        # Candidates across junctions, needs S-Rule fixing
        junctions = set(node for node, attr in self.web.nodes(data=True) if attr['type'] == 'junction')
        
        for junction in junctions:
            subwebs_across_junction = self.candidates_across_junction(junction)
            subwebs_across_junction = self.minimal_subwebs(subwebs_across_junction)
            subwebs_across_junction = self.srule_corrections(subwebs_across_junction)

            subwebs.extend(subwebs_across_junction)

        #Removing disconnected pieces, not sure how they appear in the first place but fine..
        #for subweb in subwebs:
            #subweb_graph = BraneWeb.from_subgraph_edges(self.web, subweb)
            #if nx.is_connected(subweb_graph.web): continue
            #subwebs.remove(subweb)

        # Pieces that are solely between 7-branes
        seven_seven_branes = [
            (u, v) for u, v in self.web.edges() 
            if self.web.nodes[u]['type'] == 'seven-brane' and self.web.nodes[v]['type'] == 'seven-brane'
        ]
        seven_seven_branes_counts = collections.Counter(seven_seven_branes)
        for (u, v), count in seven_seven_branes_counts.items():
            subwebs.append([(u,v)])

        return subwebs

    def srule_corrections(self, subwebs: list[Subweb]) -> list[Subweb]:
        '''Applies S-Rule corrections to a list of candidate subwebs.'''

        corrected_subwebs = []
        for subweb in subwebs:
            updated_subweb = subweb.copy()
            srule_satisfied = True

            sj_branes = []
            for u, v in subweb:
                if self.web.nodes[u]['type'] == 'junction' and self.web.nodes[v]['type'] == 'seven-brane': sj_branes.append((u, v))
                if self.web.nodes[v]['type'] == 'junction' and self.web.nodes[u]['type'] == 'seven-brane': sj_branes.append((v, u))

            sj_branes_counts = collections.Counter(sj_branes)
            for brane, count in sj_branes_counts.items():
                NS5_charge = self.NS5_charge_seen_from(brane, sj_branes)

                if NS5_charge == 0: continue
                if count > NS5_charge: 
                    excess = count - NS5_charge
                    seven_brane = brane[1] if self.web.nodes[brane[0]]['type'] == 'junction' else brane[0]
                    junction = brane[0] if self.web.nodes[brane[0]]['type'] == 'junction' else brane[1]
                    extensions = self.find_extensions((junction, seven_brane), count, NS5_charge)

                    if len(extensions) == 0:
                        srule_satisfied = False
                        break
                    updated_subweb.extend(extensions)

            if srule_satisfied:
                corrected_subwebs.append(updated_subweb)

        return corrected_subwebs

    def find_extensions(self, brane: Brane, count: int, NS5_charge: int) -> list[Brane]:
        '''Finds possible extensions for a given brane to satisfy the S-Rule. brane needs to be (junction, seven-brane)'''
        excess = count - NS5_charge
        if excess == 0: return []
    
        possible_extensions = self.web.edges(brane[1])
        extensions_on_other_side = []

        for u, v in possible_extensions:
            if v == brane[0]: continue
            if u != brane[1]: continue

            extensions_on_other_side.append((u, v))

        if len(extensions_on_other_side) < excess:
            return []

        if len(extensions_on_other_side) <= NS5_charge:
            return extensions_on_other_side[:excess]

        return self.find_extensions(extensions_on_other_side[0], count - NS5_charge, NS5_charge) + extensions_on_other_side[:excess]

    def NS5_charge_seen_from(self, brane: Brane, branes: Subweb) -> Subweb:
        '''Transforms all branes such that the given brane is a (1,0) brane and reads off the NS5 charge.'''    
        charge = np.array(self.web.edges[(brane[0], brane[1], 0)]['charge'])
        a, b = self.extended_euclidean_algorithm(charge[0], charge[1])

        transformation_matrix = np.array([[a, b], [-charge[1], charge[0]]])

        NS5_charge = 0
        for brane in branes:
            orginal_charge = np.array(self.web.edges[(brane[0], brane[1], 0)]['charge'])
            new_charge = transformation_matrix @ orginal_charge

            NS5_charge += abs(new_charge[1])
        
        return NS5_charge // 2

    def minimal_subwebs(self, subwebs: list[Subweb]) -> list[Subweb]:
        '''Filters out non-minimal subwebs.'''

        minimal_subwebs = subwebs.copy()

        for i, subweb1 in enumerate(subwebs):
            for j, subweb2 in enumerate(subwebs):
                if i == j: continue
                if len(subweb1) >= len(subweb2): continue
                if not self.is_subweb(subweb1, subweb2): continue
                if subweb2 not in minimal_subwebs: continue

                minimal_subwebs.remove(subweb2)

        # all_combinations = self.all_unions_of_subwebs(subwebs)
        # for combination in all_combinations:
            # for subweb in subwebs:
                # if not self.same_subweb(combination, subweb): continue
                # if subweb not in minimal_subwebs: continue

                # minimal_subwebs.remove(subweb)
                # break

        return minimal_subwebs

    def candidates_across_junction(self, junction: str) -> list[Subweb]:
        '''Finds all subwebs across a given junction.'''

        branes = list(self.web.edges(junction))

        candidates = []

        all_combinations = self.all_combinations_of_branes(branes, min_r=2)
        for combination in all_combinations:

            if combination in candidates: continue
            if not self.conserves_charge(combination): continue

            candidates.append(combination)

        return candidates

    def extended_euclidean_algorithm(self, p, q):
        '''Iterative implementation of the Extended Euclidean Algorithm.'''

        if p == 0:
            return (0, 1)
        
        # Initialize variables for the algorithm
        old_r, r = p, q
        old_s, s = 1, 0
        old_t, t = 0, 1
        
        while r != 0:
            quotient = old_r // r
            
            # Update r (remainder)
            old_r, r = r, old_r - quotient * r
            
            # Update s (coefficient for p)
            old_s, s = s, old_s - quotient * s
            
            # Update t (coefficient for q)
            old_t, t = t, old_t - quotient * t
            
        # gcd is old_r, and coefficients are old_s and old_t
        return old_s, old_t

    def conserves_charge(self, branes: Subweb) -> bool:
        '''Checks if a set of branes conserves charge at each junction.'''

        junctions = set()
        for u, v in branes:
            if self.web.nodes[u]['type'] == 'junction': junctions.add(u)
            if self.web.nodes[v]['type'] == 'junction': junctions.add(v)

        for junction in junctions:
            branes_at_junction = [brane for brane in branes if junction in brane]

            charge = [0, 0]
            for u, v in branes_at_junction:
                other_node = v if u == junction else u
                
                charge[0] += self.charge_into_node(junction, other_node)[0]
                charge[1] += self.charge_into_node(junction, other_node)[1]
            
            if charge != [0, 0]:
                return False

        return True

    def charge_into_node(self, node: str, other_node: str) -> tuple[int, int]:
        '''Calculates the (p,q) charge of a brane going into a node from another node.'''

        if node not in self.web.nodes or other_node not in self.web.nodes:
            raise ValueError("Both nodes must exist in the brane web.")
        
        charge = (
            self.web.nodes[node]['pos'][0] - self.web.nodes[other_node]['pos'][0],
            self.web.nodes[node]['pos'][1] - self.web.nodes[other_node]['pos'][1]
        )
        gcd = math.gcd(charge[0], charge[1])
        charge = (charge[0] // gcd, charge[1] // gcd)

        return charge

    def all_combinations_of_branes(self, branes: Subweb, min_r: int = 2) -> list[Subweb]:
        '''Generates all combinations of branes of size at least min_r.'''

        # It should be possible to reduce the set of branes for a given r to those that are not duplicates, or rather for 
        # r=2 no counts >1, for r=3 no counts >2, etc. But this is not implemented yet.
        # This may reduce the number of combinations to check significantly.

        all_combinations = []
        for r in range(min_r, len(branes) + 1):
            combinations_of_size_r = itertools.combinations(branes, r)
            all_combinations.extend(combinations_of_size_r)

        all_combinations = [list(comb) for comb in all_combinations]

        return all_combinations

    def all_unions_of_subwebs(self, subwebs: list[Subweb]) -> list[Subweb]:
        '''Generates all unions of subwebs.'''

        all_unions = []
        for r in range(2, len(subwebs) + 1):
            combinations_of_size_r = itertools.combinations(subwebs, r)

            for combination in combinations_of_size_r:
                union = []
                for subweb in combination:
                    union.extend(subweb)
                all_unions.append(union)

        return all_unions

    def same_subweb(self, subweb1: Subweb, subweb2: Subweb) -> bool:
        '''Checks if two subwebs are the same irrespective of order.'''

        sorted1 = []
        for brane in subweb1:
            sorted1.append(sorted(brane))
        sorted1 = sorted(sorted1)

        sorted2 = []
        for brane in subweb2:
            sorted2.append(sorted(brane))
        sorted2 = sorted(sorted2)

        return sorted1 == sorted2

    def same_decomposition(self, decomposition1: list[Subweb], decomposition2: list[Subweb]) -> bool:
        '''Checks if two decompositions are the same'''

        if len(decomposition1) != len(decomposition2): return False

        sorted_decomposition1 = []
        for subweb1 in decomposition1:
            sorted1 = []
            for brane in subweb1:
                sorted1.append(sorted(brane))
            sorted1 = sorted(sorted1)
            sorted_decomposition1.append(sorted1)

        sorted_decomposition2 = []
        for subweb2 in decomposition2:
            sorted2 = []
            for brane in subweb2:
                sorted2.append(sorted(brane))
            sorted2 = sorted(sorted2)
            sorted_decomposition2.append(sorted2)

        for subweb1 in sorted_decomposition1:
            if subweb1 not in sorted_decomposition2:
                return False

        return True

    def is_subweb(self, subweb1: Subweb, subweb2: Subweb) -> bool:
        '''Checks if subweb1 is contained in subweb2.'''

        sorted1 = []
        for brane in subweb1:
            sorted1.append(tuple(sorted(brane)))
        sorted1 = sorted(sorted1)

        sorted2 = []
        for brane in subweb2:
            sorted2.append(tuple(sorted(brane)))
        sorted2 = sorted(sorted2)

        sorted1_counts = collections.Counter(sorted1)
        sorted2_counts = collections.Counter(sorted2)

        for brane in sorted1_counts:
            if brane not in sorted2_counts.keys(): return False
            if sorted1_counts[brane] > sorted2_counts[brane]: return False

        return True

    def subtract_subweb(self, subweb1: Subweb, subweb2: Subweb) -> Subweb:
        '''Subtracts subweb 2 from subweb1, ie. removes relevent edges.'''
        sorted1 = []
        for brane in subweb1:
            sorted1.append(sorted(brane))
        sorted1 = sorted(sorted1)

        sorted2 = []
        for brane in subweb2:
            sorted2.append(sorted(brane))
        sorted2 = sorted(sorted2)

        for brane in sorted2:
            # if not brane in sorted1: continue
            sorted1.remove(brane)

        return sorted1

    def draw(self, save_path: str = None):
        '''Visualises the brane web.'''
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

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




