import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.strtree import STRtree
from typing import List, Optional, Tuple


class Brane:
    '''
    A single brane represented as a line segment in 2D, determined by a start point,
    a (p, q) charge vector, and a length. All inputs are numpy arrays for consistency.
    '''

    def __init__(self, pos1: np.ndarray, pq_charge: np.ndarray, length: float):
        assert pq_charge[0] != 0 or pq_charge[1] != 0, "Brane must have a non-zero (p, q) charge."
        assert length > 0, "Length must be a positive number."

        self.length = length
        self.pq_charge = np.asarray(pq_charge, dtype=int)

        self.pos1 = np.asarray(pos1, dtype=float)
        self.pos2 = pos1 + (self.length * self.pq_charge / np.linalg.norm(self.pq_charge))

        self.line = LineString([self.pos1, self.pos2])

    def plot(self, ax: Optional[plt.Axes] = None, **kwargs):
        '''
        Plot this brane segment on the given Axes.
        '''
        if ax is None:
            ax = plt.gca()

        xs, ys = self.line.xy
        ax.plot(xs, ys, **kwargs)

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

    def intersection_number(self, other: 'BraneWeb') -> int:
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

        return count
    
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

    def intersection_number(self, i: int, j: int) -> int:
        '''
        Returns the number of intersections between the web at index i and the web at index j.
        '''
        return self.brane_webs[i].intersection_number(self.brane_webs[j])

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
    
