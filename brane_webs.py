# Needs redoing


import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.strtree import STRtree
from typing import List, Optional, Tuple


class D7Brane:
    '''
    D7 brane which acts as end point for other branes.
    '''

    def __init__(self):
        self.branes_inside: List[Brane] = []
        self.branes_outside: List[Brane] = []

    def add_brane_inside(self, brane: 'Brane'):
        '''
        Add a brane to the inside of this D7 brane.
        '''
        self.branes_inside.append(brane)

    def add_brane_outside(self, brane: 'Brane'):
        '''
        Add a brane to the outside of this D7 brane.
        '''
        self.branes_outside.append(brane)

    def __hash__(self) -> int:
        # Hash by identity so D7Brane can be used in sets/dicts
        return id(self)

    def __eq__(self, other: object) -> bool:
        # Equal only if same object
        return self is other


class Brane:
    '''
    A single brane represented as a line segment in 2D, determined by a start point,
    a (p, q) charge vector, and a length. All inputs are numpy arrays for consistency.
    '''

    def __init__(
            self, 
            pos1: np.ndarray, 
            pq_charge: np.ndarray, 
            length: float, 
            starts_on: D7Brane = None,
            starts_on_side: str = None,
            ends_on: D7Brane = None,
            ends_on_side: str = None
        ):

        assert pq_charge[0] != 0 or pq_charge[1] != 0, "Brane must have a non-zero (p, q) charge."
        assert length > 0, "Length must be a positive number."
        assert starts_on_side in [None, 'inside', 'outside'], "Invalid start side specification."
        assert ends_on_side in [None, 'inside', 'outside'], "Invalid end side specification."

        self.length = length
        self.pq_charge = np.asarray(pq_charge, dtype=int)

        self.pos1 = np.asarray(pos1, dtype=float)
        self.pos2 = pos1 + (self.length * self.pq_charge / np.linalg.norm(self.pq_charge))

        self.line = LineString([self.pos1, self.pos2])

        self.starts_on = starts_on
        self.starts_on_side = starts_on_side

        self.ends_on = ends_on
        self.ends_on_side = ends_on_side

        # Adding this brane to the D7 branes on which it starts/ends
        if starts_on is not None:
            if starts_on_side == 'inside':
                starts_on.add_brane_inside(self)
            if starts_on_side == 'outside':
                starts_on.add_brane_outside(self)
        
        if ends_on is not None:
            if ends_on_side == 'inside':
                ends_on.add_brane_inside(self)
            if ends_on_side == 'outside':
                ends_on.add_brane_outside(self)

    def plot(self, ax: Optional[plt.Axes] = None, **kwargs):
        '''
        Plot this brane segment on the given Axes.
        '''
        if ax is None:
            ax = plt.gca()

        xs, ys = self.line.xy
        line_obj = ax.plot(xs, ys, **kwargs)
        
        if self.starts_on is not None:
            ax.scatter(*self.pos1, color='black', label='Start', s=20, zorder=5)

        if self.ends_on is not None:
            ax.scatter(*self.pos2, color='black', label='End', s=20, zorder=5)

        mid = (self.pos1 + self.pos2) / 2.0
        dir_vec = self.pos2 - self.pos1
        perp = np.array([-dir_vec[1], dir_vec[0]])

        if np.linalg.norm(perp) > 0:
            perp = perp / np.linalg.norm(perp) * 0.05 * self.length

        label_pos = mid + perp
        color = 'black'

        ax.text(label_pos[0], label_pos[1],
                f"({self.pq_charge[0]},{self.pq_charge[1]})",
                fontsize=9, color=color,
                ha='center', va='center')

        return ax


class BraneWeb:
    '''
    A collection of Branes, with a spatial index for fast intersection tests.
    '''

    def __init__(self, branes: List[Brane] = None):
        self.branes: List[Brane] = branes if branes is not None else []
        self._rebuild_index()

    def add_brane(self, brane: Brane):
        self.branes.append(brane)
        self._rebuild_index()

    def _rebuild_index(self):
        '''
        Rebuilds the STRtree spatial index on all current brane line segments.
        '''

        self.lines = [b.line for b in self.branes]
        self._tree = STRtree(self.lines) if self.lines else None

    def intersects(self, other: 'BraneWeb') -> bool:
        '''
        Returns True if any brane in this web intersects any brane in the other web.
        Uses STRtree to only test bounding-box-overlapping candidates.
        '''

        if not self.lines or not other.lines:
            return False
        
        for line in self.lines:
            candidates = other._tree.query(line)

            for cand in candidates:
                cand_line = other.lines[cand]

                if line.intersects(cand_line):
                    return True

        return False

    def edges(self, other: 'BraneWeb') -> int:
        '''
        Returns the number of intersections number between this web and another.
        '''

        count = 0
        for idx, line in enumerate(self.lines):
            candidates = other._tree.query(line)

            for cand in candidates:
                cand_line = other.lines[cand]

                if line.intersects(cand_line):
                    brane1 = self.branes[idx]
                    brane2 = other.branes[cand]

                    # if brane1.pq_charge[0] * brane2.pq_charge[1] - brane1.pq_charge[1] * brane2.pq_charge[0] != 0:
                    count += np.abs(brane1.pq_charge[0] * brane2.pq_charge[1] - brane1.pq_charge[1] * brane2.pq_charge[0])

        d7_branes1 = self.get_D7_branes()
        d7_branes2 = other.get_D7_branes()
        shared_d7_branes = set(d7_branes1) & set(d7_branes2)

        for d7_brane in shared_d7_branes:
            count_inside_web1 = 0
            count_outside_web1 = 0
            count_inside_web2 = 0
            count_outside_web2 = 0

            for brane in self.branes:
                mult = math.gcd(brane.pq_charge[0], brane.pq_charge[1])
                if (brane.starts_on == d7_brane and brane.starts_on_side == 'inside') or (brane.ends_on == d7_brane and brane.ends_on_side == 'inside'):
                    count_inside_web1 += 1 * mult
                if (brane.starts_on == d7_brane and brane.starts_on_side == 'outside') or (brane.ends_on == d7_brane and brane.ends_on_side == 'outside'):
                    count_outside_web1 += 1 * mult

            for brane in other.branes:
                mult = math.gcd(brane.pq_charge[0], brane.pq_charge[1])
                if (brane.starts_on == d7_brane and brane.starts_on_side == 'inside') or (brane.ends_on == d7_brane and brane.ends_on_side == 'inside'):
                    count_inside_web2 += 1 * mult
                if (brane.starts_on == d7_brane and brane.starts_on_side == 'outside') or (brane.ends_on == d7_brane and brane.ends_on_side == 'outside'):
                    count_outside_web2 += 1 * mult

            # Each pair of inside/outside counts contributes to the intersection number
            count += count_inside_web1 * count_outside_web2
            count += count_inside_web2 * count_outside_web1
            count -= count_inside_web1 * count_inside_web2
            count -= count_outside_web1 * count_outside_web2

        return count

    def get_D7_branes(self) -> List[D7Brane]:
        '''
        Returns a list of all D7 branes that are start/end points of branes in this web.
        '''
        d7_branes = []
        for brane in self.branes:
            starts_on = brane.starts_on
            ends_on = brane.ends_on

            if starts_on is not None and starts_on not in d7_branes:
                d7_branes.append(starts_on)
            if ends_on is not None and ends_on not in d7_branes:
                d7_branes.append(ends_on)
        
        return d7_branes
    
    def plot(self, ax: Optional[plt.Axes] = None, **kwargs):
        '''
        Plot all branes in this web on the given Axes.
        '''

        if ax is None:
            ax = plt.gca()

        for brane in self.branes:
            brane.plot(ax, **kwargs)

        return ax



class SuperBraneWeb:
    '''
    A container for multiple BraneWebs, with utility to test pairwise intersections.
    '''

    def __init__(self, brane_webs: List[BraneWeb] = None):
        self.brane_webs: List[BraneWeb] = brane_webs if brane_webs is not None else []

    def add_brane_web(self, brane_web: BraneWeb):
        self.brane_webs.append(brane_web)

    def webs_intersect(self, i: int, j: int) -> bool:
        '''
        Check whether the web at index i intersects the web at index j.
        '''
        return self.brane_webs[i].intersects(self.brane_webs[j])

    def edges(self, i: int, j: int) -> int:
        '''
        Returns the number of intersections between the web at index i and the web at index j.
        '''
        return self.brane_webs[i].edges(self.brane_webs[j])

    def plot(self, figsize: Tuple[int, int] = (6, 6), colors: Optional[List[str]] = None) -> plt.Figure:
        '''
        Quick plot of all webs in the super-web, each in a different color.
        '''
        
        if colors is None:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, ax = plt.subplots(figsize=figsize)

        for idx, web in enumerate(self.brane_webs):
            color = colors[idx % len(colors)]
            web.plot(ax, color=color, linewidth=2)

        ax.set_aspect('equal', 'box')

        return fig

    def __len__(self) -> int:
        return len(self.brane_webs)
    
