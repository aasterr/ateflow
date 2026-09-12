"""Declarative DAG, d-separation, and identification of causal effects.

No external dependencies: stdlib only.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

MAX_EXPLAINED_PATHS = 12
MAX_FRONTDOOR_SIZE = 3


class CyclicGraphError(ValueError):
    pass


def _as_set(nodes: str | Iterable[str]) -> set[str]:
    return {nodes} if isinstance(nodes, str) else set(nodes)


class DAG:
    """Directed acyclic graph defined by its edges.

    Text format accepted by `DAG.parse` (one relation per line):

        # comment
        crowding -> speed
        crowding -> led
        led -> speed
        unmeasured: motivation

    Node names may contain spaces. An `unmeasured:` line lists variables that
    are part of the causal story but have no column in the data: they can
    never be adjusted for, and they are what makes front-door identification
    necessary.
    """

    def __init__(self, edges: Iterable[tuple[str, str]] = (), unmeasured: Iterable[str] = ()):
        self._parents: dict[str, set[str]] = {}
        self._children: dict[str, set[str]] = {}
        self.unmeasured: set[str] = set()
        for src, dst in edges:
            self.add_edge(src, dst)
        for node in unmeasured:
            self.mark_unmeasured(node)

    # ---------- construction ----------

    def add_node(self, node: str) -> None:
        self._parents.setdefault(node, set())
        self._children.setdefault(node, set())

    def mark_unmeasured(self, node: str) -> None:
        self.add_node(node)
        self.unmeasured.add(node)

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
            if line.lower().startswith("unmeasured:"):
                for name in line.split(":", 1)[1].split(","):
                    if name.strip():
                        dag.mark_unmeasured(" ".join(name.split()))
                continue
            if "->" not in line:
                # isolated node (e.g. a disconnected variable); names may contain
                # spaces, as CSV headers often do
                dag.add_node(" ".join(line.split()))
                continue
            chain = [" ".join(tok.split()) for tok in line.split("->")]
            if any(not tok for tok in chain):
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
    def observed(self) -> set[str]:
        return self.nodes - self.unmeasured

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

    def without_outgoing(self, nodes: str | Iterable[str]) -> "DAG":
        """Manipulated graph G_{\\bar{X}}: removes the edges leaving `nodes`."""
        cut = _as_set(nodes)
        sub = DAG([(s, d) for s, d in self.edges if s not in cut], self.unmeasured)
        for n in self.nodes:
            sub.add_node(n)
        return sub

    # ---------- d-separation ----------

    def d_separated(self, x: str | Iterable[str], y: str | Iterable[str],
                    given: Iterable[str] = ()) -> bool:
        """True if (every node of) x and y are d-separated given `given`.

        Implemented by moralization of the ancestral subgraph, which is
        equivalent to d-separation and easy to verify by hand.
        """
        xs, ys, z = _as_set(x), _as_set(y), set(given)
        if (xs | ys) & z:
            raise ValueError("x and y cannot appear in the conditioning set")
        relevant = xs | ys | z
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

        stack, seen = list(xs), set(xs)
        while stack:
            cur = stack.pop()
            if cur in ys:
                return False
            for nxt in undirected[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return True

    # ---------- paths, for explanations ----------

    def paths(self, x: str, y: str, limit: int = MAX_EXPLAINED_PATHS) -> list[list[tuple[str, str]]]:
        """Simple undirected paths from x to y, as [(node, arrow_to_next)] with arrows '->' / '<-'."""
        found: list[list[tuple[str, str]]] = []

        def walk(cur: str, visited: list[str], steps: list[tuple[str, str]]) -> None:
            if len(found) >= limit:
                return
            if cur == y:
                found.append(steps + [(y, "")])
                return
            for nxt in sorted(self._children[cur]):
                if nxt not in visited:
                    walk(nxt, visited + [nxt], steps + [(cur, "->")])
            for nxt in sorted(self._parents[cur]):
                if nxt not in visited:
                    walk(nxt, visited + [nxt], steps + [(cur, "<-")])

        walk(x, [x], [])
        return found

    def path_blocker(self, path: list[tuple[str, str]], given: Iterable[str]) -> str | None:
        """Why a path is blocked given a set: the blocking node, or None if the path is open."""
        z = set(given)
        z_and_ancestors = z | self.ancestors(z)
        for i in range(1, len(path) - 1):
            node = path[i][0]
            into_from_left = path[i - 1][1] == "->"
            into_from_right = path[i][1] == "<-"
            if into_from_left and into_from_right:  # collider
                if node not in z_and_ancestors:
                    return f"{node} (a collider left alone)"
            elif node in z:
                return node
        return None

    @staticmethod
    def path_text(path: list[tuple[str, str]]) -> str:
        return " ".join(f"{n} {a}" if a else n for n, a in path)

    # ---------- backdoor ----------

    def satisfies_backdoor(self, treatment: str, outcome: str, adjustment: Iterable[str]) -> bool:
        """Pearl's backdoor criterion for the set `adjustment` (unmeasured nodes excluded)."""
        z = set(adjustment)
        if z & (self.descendants([treatment]) | {treatment}) or outcome in z:
            return False
        if z & self.unmeasured:
            return False
        return self.without_outgoing(treatment).d_separated(treatment, outcome, z)

    def _backdoor_candidates(self, treatment: str, outcome: str) -> list[str]:
        # A valid set, when one exists, can always be found among the measured
        # ancestors of treatment and outcome: nothing else is ever needed, and
        # pruning the rest keeps the search small on wide datasets.
        forbidden = self.descendants([treatment]) | {treatment, outcome} | self.unmeasured
        return sorted(self.ancestors([treatment, outcome]) - forbidden)

    def backdoor_sets(self, treatment: str, outcome: str, max_size: int | None = None) -> list[set[str]]:
        """Valid adjustment sets among the relevant nodes, ordered by cardinality."""
        candidates = self._backdoor_candidates(treatment, outcome)
        limit = len(candidates) if max_size is None else min(max_size, len(candidates))
        valid: list[set[str]] = []
        for size in range(limit + 1):
            for combo in combinations(candidates, size):
                if self.satisfies_backdoor(treatment, outcome, combo):
                    valid.append(set(combo))
        return valid

    def minimal_backdoor_set(self, treatment: str, outcome: str) -> set[str]:
        """Smallest adjustment set. Raises if no measured set satisfies the criterion."""
        candidates = self._backdoor_candidates(treatment, outcome)
        for size in range(len(candidates) + 1):
            for combo in combinations(candidates, size):
                if self.satisfies_backdoor(treatment, outcome, combo):
                    return set(combo)
        raise ValueError(
            f"no valid backdoor set for {treatment} -> {outcome}: "
            "the effect is not identifiable from this DAG by adjustment alone"
        )

    # ---------- front-door ----------

    def satisfies_frontdoor(self, treatment: str, outcome: str, mediators: Iterable[str]) -> bool:
        """Pearl's front-door criterion for the set `mediators`.

        (1) every directed path from treatment to outcome goes through them;
        (2) no open backdoor path from the treatment to them;
        (3) every backdoor path from them to the outcome is blocked by the treatment.
        """
        m = set(mediators)
        if not m or m & ({treatment, outcome} | self.unmeasured):
            return False
        cut = DAG([(s, d) for s, d in self.edges if s not in m and d not in m])
        for n in self.nodes - m:
            cut.add_node(n)
        if outcome in cut.descendants([treatment]):
            return False
        if not self.without_outgoing(treatment).d_separated(treatment, m):
            return False
        return self.without_outgoing(m).d_separated(m, outcome, {treatment})

    def frontdoor_sets(self, treatment: str, outcome: str,
                       max_size: int = MAX_FRONTDOOR_SIZE) -> list[set[str]]:
        candidates = sorted(
            (self.descendants([treatment]) & self.ancestors([outcome])) - {treatment, outcome}
            - self.unmeasured
        )
        valid: list[set[str]] = []
        for size in range(1, min(max_size, len(candidates)) + 1):
            for combo in combinations(candidates, size):
                if self.satisfies_frontdoor(treatment, outcome, combo):
                    valid.append(set(combo))
        return valid

    # ---------- the question ateflow answers ----------

    def identify(self, treatment: str, outcome: str) -> dict:
        """How the effect of treatment on outcome can be computed, and why.

        Backdoor adjustment first; front-door when no measured set closes the
        backdoor paths. Returns strategy ('backdoor' | 'frontdoor' | None),
        the variables used, alternatives, and plain-language explanation lines.
        """
        for name in (treatment, outcome):
            if name not in self.nodes:
                raise ValueError(f"{name!r} does not appear in the DAG")
            if name in self.unmeasured:
                raise ValueError(f"{name!r} is marked unmeasured: it needs a column in the data")
        if outcome not in self.descendants([treatment]):
            return {
                "strategy": "backdoor", "variables": [], "alternatives": [],
                "explanation": [f"There is no directed path from {treatment} to {outcome}: "
                                "in this DAG the causal effect is zero by assumption."],
                "backdoor_paths": [],
            }

        backdoor_paths = [p for p in self.paths(treatment, outcome) if p[0][1] == "<-"]
        try:
            minimal = sorted(self.minimal_backdoor_set(treatment, outcome))
        except ValueError:
            minimal = None

        if minimal is not None:
            explanation = []
            if not backdoor_paths:
                explanation.append(f"No backdoor path enters {treatment}: nothing confounds it, "
                                   "so the naive comparison is already causal.")
            else:
                explanation.append(
                    f"Adjusting for {{{', '.join(minimal)}}} blocks every path that enters "
                    f"{treatment} from behind (backdoor criterion):")
                for p in backdoor_paths:
                    explanation.append(f"{self.path_text(p)} — blocked at {self.path_blocker(p, minimal)}")
            alternatives = [sorted(s) for s in
                            self.backdoor_sets(treatment, outcome, max_size=len(minimal) + 1)
                            if sorted(s) != minimal]
            return {"strategy": "backdoor", "variables": minimal, "alternatives": alternatives,
                    "explanation": explanation, "backdoor_paths": [self.path_text(p) for p in backdoor_paths]}

        # the paths that defeat adjustment are the ones running through unmeasured variables
        hidden = [p for p in backdoor_paths if any(n in self.unmeasured for n, _ in p[1:-1])]
        why_not = [f"No backdoor adjustment works: {self.path_text(p)} goes through an "
                   "unmeasured variable and cannot be blocked."
                   for p in (hidden or backdoor_paths)[:3]]
        fronts = sorted(self.frontdoor_sets(treatment, outcome), key=lambda s: (len(s), sorted(s)))
        if fronts:
            m = sorted(fronts[0])
            names = ", ".join(m)
            return {
                "strategy": "frontdoor", "variables": m,
                "alternatives": [sorted(s) for s in fronts[1:4]],
                "explanation": why_not + [
                    f"Front-door through {{{names}}} instead:",
                    f"every directed path from {treatment} to {outcome} passes through {{{names}}};",
                    f"nothing confounds {treatment} and {{{names}}};",
                    f"{treatment} blocks every backdoor path from {{{names}}} to {outcome}.",
                    f"So the effect is rebuilt in two steps: {treatment} → {{{names}}}, "
                    f"then {{{names}}} → {outcome} adjusting for {treatment}.",
                ],
                "backdoor_paths": [self.path_text(p) for p in backdoor_paths],
            }
        return {
            "strategy": None, "variables": [], "alternatives": [],
            "explanation": why_not + [
                "No set of measured mediators satisfies the front-door criterion either: "
                "the effect is not identifiable from this DAG with the data available. "
                "Measuring a confounder, or a mediator that carries the whole effect, would change that."],
            "backdoor_paths": [self.path_text(p) for p in backdoor_paths],
        }

    def to_text(self) -> str:
        connected = {n for e in self.edges for n in e}
        lines = [f"{s} -> {d}" for s, d in self.edges]
        lines += sorted(n for n in self.nodes - connected - self.unmeasured)
        if self.unmeasured:
            lines.append("unmeasured: " + ", ".join(sorted(self.unmeasured)))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"DAG(nodes={len(self.nodes)}, edges={len(self.edges)})"
