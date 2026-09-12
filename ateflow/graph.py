"""Declarative DAG, d-separation, and identification of adjustment sets.

No external dependencies: stdlib only.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable


class CyclicGraphError(ValueError):
    pass


class DAG:
    """Directed acyclic graph defined by its edges.

    Text format accepted by `DAG.parse` (one relation per line):

        # comment
        crowding -> speed
        crowding -> led
        led -> speed

    Node names are strings without spaces.
    """

    def __init__(self, edges: Iterable[tuple[str, str]] = ()):
        self._parents: dict[str, set[str]] = {}
        self._children: dict[str, set[str]] = {}
        for src, dst in edges:
            self.add_edge(src, dst)

    # ---------- construction ----------

    def add_node(self, node: str) -> None:
        self._parents.setdefault(node, set())
        self._children.setdefault(node, set())

    def add_edge(self, src: str, dst: str) -> None:
        if src == dst:
            raise CyclicGraphError(f"self-loop on {src!r}")
        self.add_node(src)
        self.add_node(dst)
        self._children[src].add(dst)
        self._parents[dst].add(src)
        if self._has_cycle():
            self._children[src].discard(dst)
            self._parents[dst].discard(src)
            raise CyclicGraphError(f"edge {src} -> {dst} introduces a cycle")

    @classmethod
    def parse(cls, text: str) -> "DAG":
        dag = cls()
        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "->" not in line:
                # isolated node (e.g. a disconnected variable)
                if len(line.split()) == 1:
                    dag.add_node(line)
                    continue
                raise ValueError(f"line {lineno}: expected the form 'A -> B', found {raw!r}")
            chain = [tok.strip() for tok in line.split("->")]
            if any(not tok or " " in tok for tok in chain):
                raise ValueError(f"line {lineno}: invalid node name in {raw!r}")
            for src, dst in zip(chain, chain[1:]):
                dag.add_edge(src, dst)
        return dag

    @classmethod
    def from_file(cls, path: str) -> "DAG":
        with open(path, encoding="utf-8") as fh:
            return cls.parse(fh.read())

    # ---------- access ----------

    @property
    def nodes(self) -> set[str]:
        return set(self._parents)

    @property
    def edges(self) -> list[tuple[str, str]]:
        return [(s, d) for s, ds in self._children.items() for d in sorted(ds)]

    def parents(self, node: str) -> set[str]:
        return set(self._parents[node])

    def children(self, node: str) -> set[str]:
        return set(self._children[node])

    def ancestors(self, nodes: Iterable[str]) -> set[str]:
        return self._closure(nodes, self._parents)

    def descendants(self, nodes: Iterable[str]) -> set[str]:
        return self._closure(nodes, self._children)

    def _closure(self, nodes: Iterable[str], adj: dict[str, set[str]]) -> set[str]:
        seen: set[str] = set()
        stack = list(nodes)
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    def _has_cycle(self) -> bool:
        WHITE, GREY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self._parents}

        def visit(n: str) -> bool:
            color[n] = GREY
            for m in self._children[n]:
                if color[m] == GREY:
                    return True
                if color[m] == WHITE and visit(m):
                    return True
            color[n] = BLACK
            return False

        return any(color[n] == WHITE and visit(n) for n in list(color))

    def without_outgoing(self, node: str) -> "DAG":
        """Manipulated graph G_{\\bar{X}}: removes the edges leaving `node`."""
        kept = [(s, d) for s, d in self.edges if s != node]
        sub = DAG(kept)
        for n in self.nodes:
            sub.add_node(n)
        return sub

    # ---------- d-separation ----------

    def d_separated(self, x: str, y: str, given: Iterable[str] = ()) -> bool:
        """True if x and y are d-separated given `given`.

        Implemented by moralization of the ancestral subgraph, which is
        equivalent to d-separation and easy to verify by hand.
        """
        z = set(given)
        if x in z or y in z:
            raise ValueError("x and y cannot appear in the conditioning set")
        relevant = {x, y} | z
        keep = relevant | self.ancestors(relevant)

        # moralization: undirected edges + marrying the parents of each node
        undirected: dict[str, set[str]] = {n: set() for n in keep}
        for n in keep:
            ps = self._parents[n] & keep
            for p in ps:
                undirected[n].add(p)
                undirected[p].add(n)
            for a, b in combinations(sorted(ps), 2):
                undirected[a].add(b)
                undirected[b].add(a)

        # remove the conditioning nodes and look for a remaining path
        for n in z:
            for m in undirected.get(n, set()):
                undirected[m].discard(n)
            undirected.pop(n, None)
        if x not in undirected or y not in undirected:
            return True

        stack, seen = [x], {x}
        while stack:
            cur = stack.pop()
            if cur == y:
                return False
            for nxt in undirected[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return True

    # ---------- identification ----------

    def satisfies_backdoor(self, treatment: str, outcome: str, adjustment: Iterable[str]) -> bool:
        """Pearl's backdoor criterion for the set `adjustment`."""
        z = set(adjustment)
        if z & (self.descendants([treatment]) | {treatment}):
            return False
        if outcome in z:
            return False
        return self.without_outgoing(treatment).d_separated(treatment, outcome, z)

    def backdoor_sets(self, treatment: str, outcome: str, max_size: int | None = None) -> list[set[str]]:
        """Every valid adjustment set, ordered by cardinality."""
        forbidden = self.descendants([treatment]) | {treatment, outcome}
        candidates = sorted(self.nodes - forbidden)
        limit = len(candidates) if max_size is None else min(max_size, len(candidates))
        valid: list[set[str]] = []
        for size in range(limit + 1):
            for combo in combinations(candidates, size):
                if self.satisfies_backdoor(treatment, outcome, combo):
                    valid.append(set(combo))
        return valid

    def minimal_backdoor_set(self, treatment: str, outcome: str) -> set[str]:
        """Minimal adjustment set. Raises if the effect is not identifiable."""
        sets = self.backdoor_sets(treatment, outcome)
        if not sets:
            raise ValueError(
                f"no valid backdoor set for {treatment} -> {outcome}: "
                "the effect is not identifiable from this DAG by adjustment alone"
            )
        return min(sets, key=lambda s: (len(s), sorted(s)))

    def __repr__(self) -> str:
        return f"DAG(nodes={len(self.nodes)}, edges={len(self.edges)})"
