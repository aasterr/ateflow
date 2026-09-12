"""ateflow: causal effect estimation from a declarative DAG."""

from .api import Result, estimate_ate
from .graph import DAG

__all__ = ["DAG", "Result", "estimate_ate"]
