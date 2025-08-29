from unitary_quiver import Quiver

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

