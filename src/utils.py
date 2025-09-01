from unitary_quiver import Quiver
from brane_web import BraneWeb

import numpy as np

def init_minimal_transitions(dim_upper_bound: int = 10):
    '''Some minimal nilpotent orbit transitions... CLEAN THIS, INDICES ARE TOO MESY'''

    minimal_transitions = []

    # a_n transitions
    # ------------------------
    an_upper_bound = dim_upper_bound
    for n in range(2, an_upper_bound + 1):
        nodes = [i+1 for i in range(n+1)]
        linear_edges = [(i, i+1) for i in range(1, n)]
        all_edges = linear_edges + [(n, n+1)] + [(n+1, 1)]
        values = [1] * (n + 1)

        minimal_transitions.append(
            Quiver(nodes, all_edges, values, name=f"a_{n}")
        )

    # d_n transitions
    # ------------------------
    dn_upper_bound = np.ceil((dim_upper_bound + 3) / 2)
    for n in range(4, int(dn_upper_bound) + 1):
        nodes = [i+1 for i in range(n+1)]
        linear_edges = [(i, i+1) for i in range(1, n-1)]
        all_edges = linear_edges + [(2, n)] + [(n-2, n+1)]
        values = [1] + [2] * (n - 3) + [1] + [1, 1]

        minimal_transitions.append(
            Quiver(nodes, all_edges, values, name=f"d_{n}")
        )

    # A_n transitions
    # ------------------------
    An_upper_bound = dim_upper_bound
    for n in range(2, An_upper_bound+1):
        nodes = [1, 2]
        edges = [(1,2) for _ in range(n)]
        values = [1, 1]

        minimal_transitions.append(
            Quiver(nodes, edges, values, name=f'A_{n-1}')
        )

    # e_6 transition
    # ------------------------
    if dim_upper_bound >= 11:
        nodes = [1, 2, 3, 4, 5, 6 , 7]
        edges = [
            (1,2), (2,3), (3,4), (4,5), (3,6), (6,7)
        ]
        values = [1, 2, 3, 2, 1, 2, 1]

        minimal_transitions.append(
            Quiver(nodes, edges, values, name='e_6')
        )

    if dim_upper_bound >= 17:
        nodes = [i+1 for i in range(8)]
        edges = [(i+1, i+2) for i in range(6)] + [(4,8)]
        values = [1, 2, 3, 4, 3, 2, 1, 2]

        minimal_transitions.append(
            Quiver(nodes, edges, values, name='e_7')
        )

    if dim_upper_bound >= 29:
        nodes = [i+1 for i in range(9)]
        edges = [(i+1, i+2) for i in range(7)] + [(6,9)]
        values = [1, 2, 3, 4, 5, 6, 4, 2, 3]
        
        minimal_transitions.append(
            Quiver(nodes, edges, values, name='e_8')
        )
    
    return minimal_transitions


def init_example_branewebs() -> list:
    branewebs = []


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()

    web.add_seven_brane('7b_1', position=(-1, 0))
    web.add_seven_brane('7b_2', position=(1, 0))
    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_1', 'j1', multiplicity=1)
    web.add_brane('j1', '7b_2', multiplicity=1)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()

    web.add_seven_brane('7b_1', position=(-1, 0))
    web.add_seven_brane('7b_2', position=(1, 0))
    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_1', 'j1', multiplicity=2)
    web.add_brane('j1', '7b_2', multiplicity=2)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()

    web.add_seven_brane('7b_1', position=(-2, 0))
    web.add_seven_brane('7b_2', position=(-1, 0))
    web.add_seven_brane('7b_3', position=(1, 0))
    web.add_seven_brane('7b_4', position=(2, 0))

    web.add_junction('j1', position=(0, 0))
    web.add_brane('7b_1', '7b_2', multiplicity=1)
    web.add_brane('7b_2', 'j1', multiplicity=2)
    web.add_brane('j1', '7b_3', multiplicity=2)
    web.add_brane('7b_3', '7b_4', multiplicity=1)

    branewebs.append(web)
    
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()

    web.add_seven_brane('7b_1', position=(-3, 0))
    web.add_seven_brane('7b_2', position=(-2, 0))
    web.add_seven_brane('7b_3', position=(-1, 0))
    web.add_seven_brane('7b_4', position=(1, 0))
    web.add_seven_brane('7b_5', position=(2, 0))
    web.add_seven_brane('7b_6', position=(3, 0))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_1', '7b_2', multiplicity=1)
    web.add_brane('7b_2', '7b_3', multiplicity=2)
    web.add_brane('7b_3', 'j1', multiplicity=3)
    web.add_brane('j1', '7b_4', multiplicity=3)
    web.add_brane('7b_4', '7b_5', multiplicity=2)
    web.add_brane('7b_5', '7b_6', multiplicity=1)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()

    web.add_seven_brane('7b_1', position=(-1, 0))
    web.add_seven_brane('7b_2', position=(1, 1))
    web.add_seven_brane('7b_3', position=(0, -1))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_1', 'j1', multiplicity=1)
    web.add_brane('j1', '7b_2', multiplicity=1)
    web.add_brane('j1', '7b_3', multiplicity=1)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    
    web = BraneWeb()

    web.add_seven_brane('7b_1', position=(-1, 0))
    web.add_seven_brane('7b_2', position=(1, 1))
    web.add_seven_brane('7b_3', position=(0, -1))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_1', 'j1', multiplicity=2)
    web.add_brane('j1', '7b_2', multiplicity=2)
    web.add_brane('j1', '7b_3', multiplicity=2)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # 7

    web = BraneWeb()

    web.add_seven_brane('7b_1', position=(-1, 0))
    web.add_seven_brane('7b_2', position=(1, 0))
    web.add_seven_brane('7b_top', position=(0, 1))
    web.add_seven_brane('7b_bottom', position=(0, -1))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_1', 'j1', multiplicity=1)
    web.add_brane('j1', '7b_2', multiplicity=1)
    web.add_brane('j1', '7b_top', multiplicity=1)
    web.add_brane('j1', '7b_bottom', multiplicity=1)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # 8

    web = BraneWeb()

    web.add_seven_brane('7b_1', position=(-1, 0))
    web.add_seven_brane('7b_2', position=(1, 0))
    web.add_seven_brane('7b_top', position=(2, 1))
    web.add_seven_brane('7b_bottom', position=(-2, -1))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_1', 'j1', multiplicity=1)
    web.add_brane('j1', '7b_2', multiplicity=1)
    web.add_brane('j1', '7b_top', multiplicity=1)
    web.add_brane('j1', '7b_bottom', multiplicity=1)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()

    web.add_seven_brane('7b_1', position=(-1, 0))
    web.add_seven_brane('7b_2', position=(1, 0))

    web.add_seven_brane('7b_top', position=(1, 1))
    web.add_seven_brane('7b_bottom', position=(0, -1))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_1', 'j1', multiplicity=2)
    web.add_brane('j1', '7b_2', multiplicity=1)
    web.add_brane('j1', '7b_top', multiplicity=1)
    web.add_brane('j1', '7b_bottom', multiplicity=1)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()

    web.add_seven_brane('7b_1', position=(-2, 0))
    web.add_seven_brane('7b_2', position=(-1, 0))
    web.add_seven_brane('7b_diag', position=(2, 1))
    web.add_seven_brane('7b_bottom', position=(0, -1))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_1', '7b_2', multiplicity=1)
    web.add_brane('7b_2', 'j1', multiplicity=2)
    web.add_brane('j1', '7b_diag', multiplicity=1)
    web.add_brane('j1', '7b_bottom', multiplicity=1)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()

    web.add_seven_brane('7b_1', position=(-2, 0))
    web.add_seven_brane('7b_2', position=(-1, 0))
    web.add_seven_brane('7b_diag1', position=(1, 1))
    web.add_seven_brane('7b_diag2', position=(1, -1))
    web.add_seven_brane('7b_bottom', position=(0, -1))
    web.add_seven_brane('7b_top', position=(0, 1))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_1', '7b_2', multiplicity=1)
    web.add_brane('7b_2', 'j1', multiplicity=4)
    web.add_brane('j1', '7b_diag1', multiplicity=2)
    web.add_brane('j1', '7b_diag2', multiplicity=2)
    web.add_brane('j1', '7b_bottom', multiplicity=1)
    web.add_brane('j1', '7b_top', multiplicity=1)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()

    web.add_seven_brane('7b_1', position=(-2, 0))
    web.add_seven_brane('7b_2', position=(-1, 0))
    web.add_seven_brane('7b_diag1', position=(1, 1))
    web.add_seven_brane('7b_diag2', position=(1, -1))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_1', '7b_2', multiplicity=1)
    web.add_brane('7b_2', 'j1', multiplicity=2)
    web.add_brane('j1', '7b_diag1', multiplicity=1)
    web.add_brane('j1', '7b_diag2', multiplicity=1)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()

    web.add_seven_brane('7b_1', position=(-3, 0))
    web.add_seven_brane('7b_2', position=(-2, 0))
    web.add_seven_brane('7b_3', position=(-1, 0))
    web.add_seven_brane('7b_diag', position=(3, 1))
    web.add_seven_brane('7b_bottom', position=(0, -1))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_1', '7b_2', multiplicity=1)
    web.add_brane('7b_2', '7b_3', multiplicity=2)
    web.add_brane('7b_3', 'j1', multiplicity=3)
    web.add_brane('j1', '7b_diag', multiplicity=1)
    web.add_brane('j1', '7b_bottom', multiplicity=1)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()

    web.add_seven_brane('7b_1', position=(-2, 0))
    web.add_seven_brane('7b_2', position=(-1, 0))
    web.add_seven_brane('7b_3', position=(1, 0))
    web.add_seven_brane('7b_4', position=(2, 0))
    web.add_seven_brane('7b_diag_top', position=(-1, 1))
    web.add_seven_brane('7b_diag_bottom', position=(1, -1))
    web.add_seven_brane('7b_bottom', position=(0, -1))
    web.add_seven_brane('7b_top', position=(0, 1))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_1', '7b_2', multiplicity=1)
    web.add_brane('7b_2', 'j1', multiplicity=2)
    web.add_brane('j1', '7b_3', multiplicity=2)
    web.add_brane('7b_3', '7b_4', multiplicity=1)
    web.add_brane('j1', '7b_diag_top', multiplicity=1)
    web.add_brane('j1', '7b_diag_bottom', multiplicity=1)
    web.add_brane('j1', '7b_bottom', multiplicity=1)
    web.add_brane('j1', '7b_top', multiplicity=1)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()

    web.add_seven_brane('7b_1', position=(-3, 0))
    web.add_seven_brane('7b_2', position=(-2, 0))
    web.add_seven_brane('7b_3', position=(-1, 0))
    web.add_seven_brane('7b_4', position=(1, 0))
    web.add_seven_brane('7b_5', position=(2, 0))
    web.add_seven_brane('7b_6', position=(3, 0))

    web.add_seven_brane('7b_diag1', position=(-2, 1))
    web.add_seven_brane('7b_diag2', position=(1, 1))
    web.add_seven_brane('7b_diag3', position=(1, -1))
    web.add_seven_brane('7b_bottom', position=(0, -1))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_1', '7b_2', multiplicity=1)
    web.add_brane('7b_2', '7b_3', multiplicity=2)
    web.add_brane('7b_3', 'j1', multiplicity=3)
    web.add_brane('j1', '7b_4', multiplicity=3)
    web.add_brane('7b_4', '7b_5', multiplicity=2)
    web.add_brane('7b_5', '7b_6', multiplicity=1)
    web.add_brane('j1', '7b_diag1', multiplicity=1)
    web.add_brane('j1', '7b_diag2', multiplicity=1)
    web.add_brane('j1', '7b_diag3', multiplicity=1)
    web.add_brane('j1', '7b_bottom', multiplicity=1)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()

    web.add_seven_brane('7b_2', position=(-1, 0))
    web.add_seven_brane('7b_3', position=(1, 0))

    web.add_seven_brane('7b_bottom1', position=(0, -1))
    web.add_seven_brane('7b_bottom2', position=(1, -1))

    web.add_seven_brane('7b_top1', position=(0, 1))
    web.add_seven_brane('7b_top2', position=(-1, 1))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_2', 'j1', multiplicity=1)
    web.add_brane('j1', '7b_3', multiplicity=1)

    web.add_brane('7b_bottom1', 'j1', multiplicity=1)
    web.add_brane('7b_bottom2', 'j1', multiplicity=1)

    web.add_brane('7b_top1', 'j1', multiplicity=1)
    web.add_brane('7b_top2', 'j1', multiplicity=1)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()

    web.add_seven_brane('7b_2', position=(-1, 0))
    web.add_seven_brane('7b_3', position=(1, 0))
    web.add_seven_brane('7b_4', position=(2, 0))

    web.add_seven_brane('7b_bottom1', position=(0, -1))
    web.add_seven_brane('7b_bottom2', position=(0, -2))

    web.add_seven_brane('7b_top1', position=(0, 1))
    web.add_seven_brane('7b_top2', position=(-1, 1))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_2', 'j1', multiplicity=1)
    web.add_brane('j1', '7b_3', multiplicity=2)
    web.add_brane('7b_3', '7b_4', multiplicity=1)

    web.add_brane('7b_bottom1', 'j1', multiplicity=2)
    web.add_brane('7b_bottom1', '7b_bottom2', multiplicity=1)

    web.add_brane('7b_top1', 'j1', multiplicity=1)
    web.add_brane('7b_top2', 'j1', multiplicity=1)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()

    web.add_seven_brane('7b_1', position=(-2, 0))
    web.add_seven_brane('7b_2', position=(-1, 0))
    web.add_seven_brane('7b_3', position=(1, 0))
    web.add_seven_brane('7b_4', position=(2, 0))

    web.add_seven_brane('7b_bottom1', position=(0, -1))
    web.add_seven_brane('7b_bottom2', position=(0, -2))

    web.add_seven_brane('7b_top1', position=(0, 1))
    web.add_seven_brane('7b_top2', position=(0, 2))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_1', '7b_2', multiplicity=1)
    web.add_brane('7b_2', 'j1', multiplicity=2)
    web.add_brane('j1', '7b_3', multiplicity=2)
    web.add_brane('7b_3', '7b_4', multiplicity=1)

    web.add_brane('7b_bottom1', 'j1', multiplicity=2)
    web.add_brane('7b_bottom1', '7b_bottom2', multiplicity=1)

    web.add_brane('7b_top1', 'j1', multiplicity=2)
    web.add_brane('7b_top2', '7b_top1', multiplicity=1)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()


    web.add_seven_brane('7b_1', position=(-3, 0))
    web.add_seven_brane('7b_2', position=(-2, 0))
    web.add_seven_brane('7b_3', position=(-1, 0))

    web.add_seven_brane('7b_bottom1', position=(0, -1))
    web.add_seven_brane('7b_bottom2', position=(0, -2))
    web.add_seven_brane('7b_bottom3', position=(0, -3))

    web.add_seven_brane('7b_diag1', position=(1, 1))
    web.add_seven_brane('7b_diag2', position=(2, 2))
    web.add_seven_brane('7b_diag3', position=(3, 3))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_1', '7b_2', multiplicity=1)
    web.add_brane('7b_2', '7b_3', multiplicity=2)
    web.add_brane('7b_3', 'j1', multiplicity=3)

    web.add_brane('7b_bottom1', 'j1', multiplicity=3)
    web.add_brane('7b_bottom2', '7b_bottom1', multiplicity=2)
    web.add_brane('7b_bottom3', '7b_bottom2', multiplicity=1)

    web.add_brane('7b_diag1', 'j1', multiplicity=3)
    web.add_brane('7b_diag2', '7b_diag1', multiplicity=2)
    web.add_brane('7b_diag3', '7b_diag2', multiplicity=1)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()

    web.add_seven_brane('7b_t', position=(-3, 1))
    web.add_seven_brane('7b_b', position=(-1, -1))
    web.add_seven_brane('7b1', position=(1, 0))
    web.add_seven_brane('7b2', position=(2, 0))
    web.add_seven_brane('7b3', position=(3, 0))
    web.add_seven_brane('7b4', position=(4, 0))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_t', 'j1', multiplicity=1)
    web.add_brane('7b_b', 'j1', multiplicity=1)
    web.add_brane('j1', '7b1', multiplicity=4)
    web.add_brane('7b1', '7b2', multiplicity=3)
    web.add_brane('7b2', '7b3', multiplicity=2)
    web.add_brane('7b3', '7b4', multiplicity=1)

    branewebs.append(web)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    web = BraneWeb()

    web.add_seven_brane('7b_t', position=(-3, 1))
    web.add_seven_brane('7b_b', position=(0, -1))
    web.add_seven_brane('7b1', position=(1, 0))
    web.add_seven_brane('7b2', position=(2, 0))
    web.add_seven_brane('7b3', position=(3, 0))

    web.add_junction('j1', position=(0, 0))

    web.add_brane('7b_t', 'j1', multiplicity=1)
    web.add_brane('7b_b', 'j1', multiplicity=1)
    web.add_brane('j1', '7b1', multiplicity=3)
    web.add_brane('7b1', '7b2', multiplicity=2)
    web.add_brane('7b2', '7b3', multiplicity=1)

    branewebs.append(web)


    return branewebs
