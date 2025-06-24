




class Symmetry:
    '''
    Represents a continous symmetry group, such as U(n), SU(n), SO(n), Sp(n), etc.

    Attributes:
        type (str): The type of symmetry, e.g., 'U', 'SU', 'SO', 'Sp'.
        n (int): The matrix dimension of the symmetry group.
    '''

    def __init__(self, type: str, n: int):
        assert type in ['U', 'SU'], "Symmetry type must be one of 'U', 'SU'"
        assert n > 0, "Dimension n must be a positive integer."
        
        self.type = type
        self.n = n

    @property
    def rank(self) -> int:
        if self.type == 'U':
            return self.n
        if self.type == 'SU':
            return self.n - 1
        
        raise ValueError("Unknown symmetry type.")


class Node:
    def __init__(self, id: str, sym: Symmetry, kind: str = 'gauge'):
        assert kind in ['gauge', 'flavor'], "Node kind must be 'gauge' or 'flavor'."

        self.id = id
        self.sym = sym
        self.kind = kind

        self.excess: int = - 2*sym.rank

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __repr__(self):
        return f"Node(id={self.id}, sym={self.sym.type}({self.sym.n}), kind={self.kind})"
        

class Quiver:
    '''
    Represents a magnetic quiver, which is an undirected graph with nodes and edges.
    '''

    def __init__(self):
        self._adj = {}

    # --- Mutability methods ---
    def add_node(self, node: Node):
        if node not in self._adj:
            self._adj[node] = set()

    def add_edge(self, node1: Node, node2: Node):
        self.add_node(node1)
        self.add_node(node2)

        self._adj[node1].add(node2)
        self._adj[node2].add(node1)

        node1.excess += node2.sym.rank
        node2.excess += node1.sym.rank

    # --- Access methods ---
    def neighbors(self, node: Node):
        if node not in self._adj:
            return set()
        return self._adj[node]

    def nodes(self):
        return self._adj.keys()

    def edges(self):
        """Yield each undirected edge exactly once, as (u, v)."""
        emitted = set()                         # keeps frozensets {u, v}

        for u, nbrs in self._adj.items():
            for v in nbrs:
                edge_key = frozenset((u, v))    # {u, v} == {v, u}
                if edge_key not in emitted:
                    emitted.add(edge_key)
                    yield u, v

    # --- Plotting methods ---


