import numpy as np
from kperm.permeation import (
    _count_perm_cross,
    _partition_perm_events,
    _reduce_cycle,
    _compute_trans_prob,
    _plot_cycle,
    _compute_first_passage_times,
    _compute_mfpt,
    _plot_netflux,
)


def test_count_perm_cross_simulated_input_1():
    occ = [
        [[100], [0], [1], [], [9], [101]],
        [[0], [], [1], [], [9], [101]],
        [[0], [1], [9], [], [], [101]],
        [[100], [0], [1], [], [9], [101]],
        [[100], [0], [1], [], [9], [101]],
    ]
    output = _count_perm_cross(occ)

    assert np.array_equal(output[0], np.array([0, 0, 1, -1, 0]))
    assert output[1] == [(2, 9)]
    assert output[2] == [(3, 9)]


def test_partition_perm_events():
    occ = np.array(
        [
            "WKK0KW",
            "WKK0KW",
            "WK0KKW",
            "WKK0KW",
            "C0K0KW",
            "W0K0KW",
            "WK0K0C",
            "WKK0KW",
            "WKK0KW",
            "WK0KKW",
            "WKK0KW",
            "CK0K0C",
            "WKK0KW",
        ]
    )
    k_j = np.array([0, -1, 1, 1, 0, 2, 2, 0, -1] + [1, 3, 2])
    cycle_s = "WKK0KW"
    n_j_per_cycle = 5

    output = _partition_perm_events(occ, k_j, cycle_s, n_j_per_cycle)

    assert np.array_equal(output, np.array([[0, 7], [7, 12]]))


def test_reduce_cycle():
    occ = np.array(
        ["WKK0KW", "WKK0KW", "WK0KKW", "WKK0KW",
            "C0K0KW", "W0K0KW", "WK0K0C", "WKK0KW"]
    )

    k_j = np.array([0, -1, 1, 1, 0, 2, 2, 0])

    output = _reduce_cycle(occ, k_j)

    assert np.array_equal(
        output, np.array(["WKK0KW", "C0K0KW", "W0K0KW", "WK0K0C", "WKK0KW"])
    )


def test_compute_trans_prob_1():
    trajs = [
        np.array(["b", "b", "b", "c", "c", "a", "b", "a"]),
        np.array(["a", "b", "c", "b", "b", "b", "a", "c", "c", "c", "c", "c"]),
        np.array(["c", "c", "a", "a", "b", "a", "b", "a"]),
    ]
    output = _compute_trans_prob(trajs, quiet=False)

    assert output["c"]["c"][-1] == 6
    assert output["b"]["a"][0] == 0.4
    assert output["a"]["count"] == 8


def test_compute_trans_prob_2():
    trajs = [
        [["b", "b", "b", "c", "c", "a", "b", "a"]],
        [["a", "b", "c", "b", "b", "b", "a", "c", "c", "c", "c", "c"]],
        [["c", "c", "a", "a", "b", "a", "b", "a"]],
        [["d", "d"]],
    ]
    (
        trans_prob_dict,
        trans_prob_matrix,
        state_to_idx,
        idx_to_state,
    ) = _compute_trans_prob(trajs, quiet=True, return_matrix=True)

    assert trans_prob_dict["c"]["c"][-1] == 6
    assert trans_prob_dict["b"]["a"][0] == 0.4
    assert trans_prob_dict["a"]["count"] == 8
    assert isinstance(trans_prob_matrix, np.ndarray)
    assert trans_prob_matrix.shape == (4, 4)
    assert "d" in state_to_idx.keys()
    assert 2 in idx_to_state.keys()


def test_plot_cycle():
    cycles = [
        [np.array(["a", "b", "c", "d", "a"])],
        [np.array(["a", "c", "d"])],
        [np.array(["a", "b", "c", "d", "a"])],
    ]

    cycle_prob, main_cycle = _plot_cycle(
        cycles, figsize=(2, 2), cycle_prob=True, main_cycle=True, show=False
    )

    assert cycle_prob["a"]["prob"] == 1.0
    assert main_cycle == [
        [["a"], ["b"]],
        [["b"], ["c"]],
        [["c"], ["d"]],
        [["d"], ["a"]],
    ]

    cycle_prob = _plot_cycle(
        cycles, figsize=(2, 2), cycle_prob=True, main_cycle=False, show=False
    )

    assert cycle_prob["a"]["prob"] == 1.0


def test_compute_first_passage_times():
    occ = np.array(
        [
            "WKK0KW",
            "WKK0KW",
            "WK0KKW",
            "WKK0KW",
            "C0K0KW",
            "W0K0KW",
            "WK0K0C",
            "WKK0KW",
            "WKK0KW",
            "WK0KKW",
            "WKK0KW",
            "CK0K0C",
            "WKK0KW",
        ]
    )

    k_j = np.array([0, -1, 1, 1, 0, 2, 2, 0, -1, 1, 3, 2])
    k_w = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    jumps = np.vstack((k_j, k_w)).T

    output = _compute_first_passage_times(
        occ, jumps, ["WKK0KW"], ["WK0KKW"], backward=True
    )

    assert output == ([2], [-1], [0])

    output = _compute_first_passage_times(
        occ, jumps, ["WKK0KW"], ["WK0KKW"], backward=False
    )

    assert output == ([6], [4], [0])

    output = _compute_first_passage_times(
        occ, None, ["WKK0KW"], ["WK0KKW"], backward=False
    )

    assert output == ([2, 6], [0, 0], [0, 0])


def test_compute_mfpt():
    trajs = [
        np.array(
            ["WKK0KW", "WKK0KW", "C0K0KW", "WKK0KW", "C0K0KW",
             "WKK0KW", "C0K0KW"]
        ),
        np.array(
            ["WKK0KW", "WKK0KW", "C0K0KW", "WKK0KW", "C0K0KW",
             "WKK0KW", "C0K0KW"]
        ),
        np.array(
            ["WKK0KW", "WKK0KW", "C0K0KW", "WKK0KW", "C0K0KW",
             "WKK0KW", "C0K0KW"]
        ),
    ]

    k_j = np.array([0, 1, -1, 1, -1, 1, 0])
    k_w = np.array([0, 0, 0, 0, 0, 0, 0])
    jumps = np.vstack((k_j, k_w)).T

    jumps_all = [jumps, jumps, jumps]
    df, fpts = _compute_mfpt(
        trajs, jumps_all, [[["WKK0KW"], ["C0K0KW"]]], dt=0.75)

    assert all(np.equal(df["mfpt"].to_numpy(), 1.0))
    assert all(np.equal(df["n"].to_numpy(), 9))
    assert all(np.equal(df["k_f"].to_numpy(), 1.0))
    assert all(
        np.equal(
            fpts["WKK0KW-C0K0KW"],
            np.array([1.5, 0.75, 0.75, 1.5, 0.75, 0.75, 1.5, 0.75, 0.75]),
        )
    )

    trajs = [
        np.array(["WKK0KW", "WKK0KW", "C0K0KW"]),
        np.array(["WKK0KW", "WKK0KW", "C0K0KW"]),
        np.array(["WKK0KW", "WKK0KW", "C0K0KW"]),
    ]

    k_j = np.array([0, 1, -1])
    k_w = np.array([0, 0, 0])
    jumps = np.vstack((k_j, k_w)).T

    jumps_all = [jumps, jumps, jumps]
    df, fpts = _compute_mfpt(
        trajs, jumps_all,
        [[["WKK0KW"], ["C0K0KW"]], [["WKK0KW"], ["C0K0KQ"]]],
        dt=0.75
    )

    assert df["mfpt"][0] == 1.5
    assert df["mfpt_l"][0] == 0.0
    assert df["mfpt_h"][0] == 0.0

    assert df["mfpt"][1] == 0.0
    assert df["mfpt_l"][1] == 0.0
    assert df["mfpt_h"][1] == 0.0


def test_plot_netflux():
    trajs = [
        np.array(["a", "b", "c", "d", "a"]),
        np.array(["b", "c", "d", "a", "b"]),
        np.array(["c", "d", "a", "b", "c"]),
        np.array(["d", "a", "b", "c", "d"]),
    ]

    state_probs, edge_weights = _plot_netflux(trajs, data=True, show=False)

    assert state_probs["b"] == 0.25
    assert edge_weights[("a", "b")] == 0.25
