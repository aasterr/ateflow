"""Backdoor criterion tests on canonical structures."""

import pytest

from ateflow import DAG


def test_confounder_is_selected():
    dag = DAG.parse("z -> t\nz -> y\nt -> y")
    assert dag.minimal_backdoor_set("t", "y") == {"z"}


def test_collider_must_not_be_adjusted():
    # t -> c <- y : conditioning on c opens a spurious path
    dag = DAG.parse("t -> c\ny -> c\nt -> y")
    assert dag.minimal_backdoor_set("t", "y") == set()
    assert not dag.satisfies_backdoor("t", "y", ["c"])


def test_mediator_is_excluded():
    # t -> m -> y : m is a descendant of the treatment
    dag = DAG.parse("t -> m -> y")
    assert not dag.satisfies_backdoor("t", "y", ["m"])


def test_descendant_of_treatment_is_excluded():
    dag = DAG.parse("z -> t\nz -> y\nt -> y\nt -> w")
    assert not dag.satisfies_backdoor("t", "y", ["z", "w"])


def test_unidentifiable_without_confounder():
    # u non osservato ma dichiarato nel DAG: l'insieme {u} resta l'unico valido
    dag = DAG.parse("u -> t\nu -> y\nt -> y")
    assert dag.minimal_backdoor_set("t", "y") == {"u"}


def test_m_bias_structure():
    # a -> t, a -> m, b -> m, b -> y : aggiustare per m introduce M-bias
    dag = DAG.parse("a -> t\na -> m\nb -> m\nb -> y\nt -> y")
    assert dag.minimal_backdoor_set("t", "y") == set()
    assert not dag.satisfies_backdoor("t", "y", ["m"])


def test_d_separation_chain_and_fork():
    dag = DAG.parse("a -> b -> c")
    assert dag.d_separated("a", "c", ["b"])
    assert not dag.d_separated("a", "c")
    fork = DAG.parse("b -> a\nb -> c")
    assert fork.d_separated("a", "c", ["b"])


def test_cycle_is_rejected():
    with pytest.raises(Exception):
        DAG.parse("a -> b\nb -> a")
