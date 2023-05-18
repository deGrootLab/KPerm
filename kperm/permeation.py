"""methods for computing permeation cycles

The module contains the methods used by class Channel for computing
permeation cycles.

"""
import numpy as np
import scipy.stats

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def _count_perm_cross(occupancy, group1=(0, 1, 2), group2=(3, 4, 5)):
    """Identify permeation events crossing the S2/S3 plane

    Parameters
    ----------
    occupancy: array of size N
        tranjectory expressed in the form of SF occupancy

    group1: list of ints
        indicies of binding sites defining the plane on the extracelluar side

    group2: list of ints
        indicies of binding sites defining the plane on the intracelluar side

    Returns
    -------
    perm_traj: narray
        array of the same shape as occupancy, containing net number of
        permeation events in each frame t. perm_traj[t] > 0 refers to
        a net number of upward permeation events at time t.
    perm_up: list of tuples
        list of the same length as occupancy. Each tuple (t, i) contains
        the time t at which a permeant object of the index i crosses the plane
        upward, towards the extracellular side.
    perm_down: list of tuples
        list of the same length as occupancy. Each tuple (t, i) contains
        the time t at which a permeant object of the index i crosses the plane
        downward, towards the intracellular side.
    """
    perm_traj = np.zeros(len(occupancy), dtype=int)
    permeating_objects = dict()
    perm_up = []
    perm_down = []

    for t, occ in enumerate(occupancy):
        for site in group1:
            for index in occ[site]:
                if index not in permeating_objects:
                    permeating_objects[index] = 1
                elif permeating_objects[index] == 1:
                    continue
                elif permeating_objects[index] == 2:
                    # from group 2 to group 1, so it's an upward flow
                    permeating_objects[index] = 1
                    perm_traj[t] += 1

                    perm_up.append((t, index))

        for site in group2:
            for index in occ[site]:
                if index not in permeating_objects:
                    permeating_objects[index] = 2
                elif permeating_objects[index] == 2:
                    continue
                elif permeating_objects[index] == 1:
                    # from group 1 to group 2, so it's a downward flow
                    permeating_objects[index] = 2
                    perm_traj[t] -= 1

                    perm_down.append((t, index))

        # iterate over permeating_objects to check if any of the bound objects
        # escape from SF, remove them if they have escaped
        indices = list(permeating_objects.keys())
        for index in indices:
            if any([index in occ[site] for site in group1]) or any(
                [index in occ[site] for site in group2]
            ):
                continue
            else:
                permeating_objects.pop(index)

    return perm_traj, perm_up, perm_down


def _partition_perm_events(occupancy, k_jump, cycle_state, n_jump_per_cycle=5):
    """partitioning trajectory into full permeation cycles

    Parameters
    ----------
    occupancy: array of size (N, )
        tranjectory expressed in the form of SF occupancy

    k_jump: arrays of size (N, )
        net jumps for ions in the associated cycle_original

    cycle_state: string
        the SF occupation state that the cycles start and end in

    n_jump_per_cycle: int
        # of ions that constitute a complete permeation cycle.
        Example:
            n_jump_per_cycle = 5 if only jumps from or to S1, S2, S3, and S4
            are considered.

    Returns
    -------
    perm_indices: array of int of size (N, 2)
        index for the start and the end of permeation cycles
    """

    assert n_jump_per_cycle > 0, "n_jump_per_cycle should be > 0"

    L = len(occupancy)

    t = 0
    T = 0
    found = False

    perm_indices = []

    while t < L - 1:
        if occupancy[t] == cycle_state:
            T = 0
            found = False

            while T < L - t - 1 and found is False:
                T = T + 1

                if occupancy[t + T] == cycle_state:
                    n_k_jump = np.sum(k_jump[t: t + T])

                    if n_k_jump == n_jump_per_cycle:
                        perm_index_pair = [t, t + T]
                        perm_indices.append(perm_index_pair)
                        found = True
                    elif (
                        n_k_jump <= -n_jump_per_cycle
                        or n_k_jump >= 2 * n_jump_per_cycle
                    ):
                        found = True
                        print(
                            f"Not returning to {cycle_state}, \
no cycle is formed"
                        )
            t = t + T - 1
        t = t + 1

    perm_indices = np.array(perm_indices)

    return perm_indices


def _reduce_cycle(cycle_original, k_jumps_sub):
    """Given an uncompressed (involving osciliation between states without net
    jumps) trajectory segment that starts and ends in the same state and
    records one complete permeation event + the associated jump vectors,
    compute the "cleaned" cycle keeping only the first hit states.

    Parameters
    ----------
    cycle_original: array of size (N, )
        Original cycle of SF occupancy representing one permeation event.
        The first and the last state has to be the same.

    k_jumps_sub: array of size (N, )
        net jumps for ion in the associated cycle_original

    Returns
    -------
    occupancy_compressed: array of size (M, )
        "cleaned" cycle keeping only the first hit states
    """
    assert (
        cycle_original[0] == cycle_original[-1]
    ), "First and last state \
not the same"

    T = len(cycle_original)

    t = 0
    cycle_reduced = []

    while t < T - 1:
        state = cycle_original[t]
        if state not in cycle_reduced:
            cycle_reduced.append(state)

            indices = np.argwhere(cycle_original == state).reshape(-1)

            for t2 in indices[::-1]:
                if np.sum(k_jumps_sub[t:t2]) == 0:
                    t = t2
                    break
        t = t + 1
    return np.array(cycle_reduced + [cycle_reduced[0]])


def _find_cycles(occupancy_all, jumps_all, cycle_state, n_jump_per_cycle=5):
    """Given occupancy and jumps of the trajectories, the seed state, and
    n_jump_per_cycle that define the # BSs in which jumps in and out are
    considered, give the cycles that start and end in the seed state.

    Parameters
    ----------
    occupancy_all: list of arrays of size N
        occupancy for all trajectories

    jumps_all: list of arrays of size (N-1, 2)
        net jumps for ion and water for all trajectories

    cycle_state: string
        the SF occupation state that the cycles start and end in

    n_jump_per_cycle: int
        # of ions that constitute a complete permeation cycle.
        Example:
            n_jump_per_cycle = 5 if only jumps from or to S1, S2, S3, and S4
            are considered.

    Returns
    -------
    permeationCycle_indices: list of lists of arrays
        contain the cycles identified in each trajectory
    """
    full_perm_cycles = []
    full_perm_idx = []

    n_k_netjumps = []
    n_identified_cycles = []

    for i, (occupancy, jumps) in enumerate(zip(occupancy_all, jumps_all)):
        print(f"Trajectory {i}")

        p_indices = _partition_perm_events(
            occupancy, jumps, cycle_state, n_jump_per_cycle=n_jump_per_cycle
        )
        full_perm_idx.append(p_indices)

        full_perm_cycles_ = [
            _reduce_cycle(occupancy[i: j + 1], jumps[i: j + 1, 0])
            for (i, j) in p_indices
        ]
        full_perm_cycles.append(full_perm_cycles_)

        n_k_netjump = np.sum(jumps[:, 0]) // n_jump_per_cycle
        n_identified_cycle = len(p_indices)

        n_k_netjumps.append(n_k_netjump)
        n_identified_cycles.append(n_identified_cycle)

        print(f"Number of permeation events: {int(n_k_netjump)}")
        print(
            f"Number of identified cycles: {n_identified_cycle} \t"
            + f"{n_identified_cycle/n_k_netjump * 100:.2f}%\n"
        )

    identified_percentage = np.sum(n_identified_cycles) / np.sum(n_k_netjumps)

    print(f"Total number of permeation events: {int(np.sum(n_k_netjumps))}")
    print(
        f"Total number of identified cycles: {np.sum(n_identified_cycles)}\t"
        + f" {identified_percentage * 100:.3f}%\n\n"
    )

    return full_perm_cycles, full_perm_idx, identified_percentage


def _compute_trans_prob(trajs, return_matrix=False, quiet=False):
    """given trajectories, compute transition probabilities

    Parameters
    ----------
    trajs: list of arrays of size N    or   list of lists of arrays
        trajectories, can be occupancy or cycles for all trajectories

    return_matrix: boolean
        if True, return also the transition matrix and mapping between indices
        and state names

    quiet: boolean
        if True, do not print anything to stdout

    Returns
    -------
    trans_prob_dict: dict
        dict containing count and prob of states, and the transition probs

    trans_prob: narray (optional)
        transition matrix

    state_to_idx: dict
        map from state name to idx

    idx_to_state: dict
        map from idx to state name (reverse of state_to_idx)

    """
    # for list of lists of arrays
    if isinstance(trajs, list) and isinstance(trajs[0], list):
        trajs = [s for traj in trajs for s in traj]

    states, counts = np.unique(np.concatenate(trajs), return_counts=True)
    sort_idx = np.argsort(counts)[::-1]
    states = states[sort_idx]
    counts = counts[sort_idx]
    counts_total = np.sum(counts)

    trans_prob_dict = {state: {} for state in states}

    state_to_idx = {s: i for i, s in enumerate(states)}
    idx_to_state = {i: s for i, s in enumerate(states)}

    n_states = len(states)
    trans_counts = np.zeros((n_states, n_states), dtype=int)

    for traj in trajs:
        for t in range(len(traj) - 1):
            i, j = state_to_idx[traj[t]], state_to_idx[traj[t + 1]]
            trans_counts[i, j] += 1

    trans_prob = np.nan_to_num(
        np.asarray(
            [
                trans_counts[i] / (np.sum(trans_counts[i]) or 1.0)
                for i in range(n_states)
            ]
        )
    )

    n_outward_total = np.sum(trans_counts, axis=1)

    for i in range(n_states):
        state_i = idx_to_state[i]
        n_total = n_outward_total[i]

        indices = np.argsort(trans_prob[i])[::-1]
        for j in indices:
            state_j = idx_to_state[j]
            p = trans_prob[i, j]
            if n_total < 2:
                err = 0.0
            else:
                err = 1.96 * np.sqrt(p * (1 - p) / (n_total - 1))
            n = trans_counts[i, j]
            trans_prob_dict[state_i][state_j] = (trans_prob[i, j], err, n)

        trans_prob_dict[state_i]["count_out"] = n_total
        trans_prob_dict[state_i]["count"] = counts[i]
        trans_prob_dict[state_i]["prob"] = counts[i] / counts_total

    if not quiet:
        for state_i, state_i_dict in trans_prob_dict.items():
            print(f"\n======== {state_i} {state_i_dict['count_out']}=========")
            for k, v in state_i_dict.items():
                if k not in ["count", "prob", "count_out"]:
                    state_j = k
                    p, err, n = v
                    if p != 0.0:
                        print(f"{state_j:<10}\t{p:<12.5%} ± {err:<12.5%} {n}")

    if return_matrix:
        return trans_prob_dict, trans_prob, state_to_idx, idx_to_state
    else:
        return trans_prob_dict


def _plot_cycle(
    cycles,
    state_threshold=0.05,
    label_threshold=0.05,
    offset=0.1,
    scale=0.1,
    figsize=(10, 10),
    save=None,
    returnCycleProb=False,
    returnMainPath=False,
):
    """given trajectories, compute transition probabilities

    Parameters
    ----------
    cycles: list of lists of arrays
        contain the cycles identified in each trajectory

    Returns
    -------
    returnCycleProb: boolean
        If True, return cycles_dict
            cycles_dict[state_i][state_j] is the probability of observing the
            transition state_i -> state_j in a permeation cycle which starts
            and ends in a specified state.

    returnMainPath: boolean
        If True, return the dominant cycle for determining MFPT using
        permeationMFPT().

    """
    # flatten nested cycles
    cycles_flattened = [c for cycle in cycles for c in cycle]
    n_cycles = len(cycles_flattened)

    # probs: dict, key=state, value=(target state, %, count)
    cycles_dict = _compute_trans_prob(cycles_flattened, quiet=True)

    ############################################################

    # find the backbone of the cyclic graph
    states_all = np.array([k for k in cycles_dict.keys()])
    state_counts = np.array([cycles_dict[state]["count_out"]
                            for state in states_all])
    state_counts_total = np.sum(state_counts)
    state_p = state_counts / state_counts_total
    states_selected = states_all[state_p > state_threshold]

    # assume that states are already sorted in descending order of population
    cycle_state = list(cycles_dict)[0]

    backbone = []
    backbone.append(cycle_state)

    state = list(cycles_dict[cycle_state].keys())[0]
    backbone.append(state)

    while backbone[-1] != cycle_state:
        state = list(cycles_dict[state].keys())[0]
        backbone.append(state)

    # assign the side branches of the cyclic graph
    non_backbone = states_selected[np.in1d(
        states_selected, backbone, invert=True)]
    sidechain = {k: [] for k in backbone}

    # put next to the backbone state before the target state which the
    # non-backbone state has highest probability transitioning to
    for state_i in non_backbone:
        j_idx = None
        for state_j in list(cycles_dict[state_i]):
            try:
                j_idx = backbone.index(state_j)
                break
            except ValueError:
                continue
        sidechain[backbone[j_idx - 1]].append(state_i)

    ############################################################
    # determine position of nodes

    # centered at origin by default
    pos = nx.circular_layout(nx.path_graph(backbone))
    for state_backbone, states_sidechain in sidechain.items():
        vec = pos[state_backbone] / np.linalg.norm(pos[state_backbone])
        for j, state_sidechain in enumerate(states_sidechain):
            pos[state_sidechain] = pos[state_backbone] + \
                offset * vec + vec * j * scale

    ############################################################
    # initialize graph
    _ = plt.figure(1, figsize=figsize, dpi=300)
    _ = plt.axis("equal")
    G = nx.DiGraph()
    G.add_nodes_from(states_selected)
    sizes = np.array([cycles_dict[state]["count_out"]
                     for state in states_selected])

    nx.draw_networkx(
        G,
        pos,
        node_color="orange",
        node_size=100 * sizes / n_cycles,
        alpha=1,
        font_size=6,
        font_color="k",
    )
    ############################################################

    for state_i in states_all:
        _ = cycles_dict[state_i].pop("count")
        cycles_dict[state_i]["prob"] = \
            cycles_dict[state_i]["count_out"] / n_cycles
        for state_j in states_all:
            # here p is defined as the probability of having this transition
            # in every observed permeation cycle, different from the
            # Markov transition prob
            _, _, count = cycles_dict[state_i][state_j]
            p = count / n_cycles
            if n_cycles < 2:
                err = 0
            else:
                err = 1.96 * np.sqrt(p * (1 - p) / (n_cycles - 1))
            cycles_dict[state_i][state_j] = (p, err, count)

    for state_i in states_selected:
        for state_j in states_selected:
            if state_i != state_j:
                p, err, _ = cycles_dict[state_i][state_j]
                alpha = np.tanh(3 * p) / np.tanh(3)
                _ = nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[(state_i, state_j, 2)],
                    arrowsize=12,
                    min_source_margin=10,
                    min_target_margin=10,
                    alpha=alpha,
                    connectionstyle="arc3,rad=0.2",
                )
                if p > label_threshold:
                    label = rf"{p*100:.1f}$\pm${err*100:.1f}%"
                    _ = nx.draw_networkx_edge_labels(
                        G,
                        pos,
                        edge_labels={(state_i, state_j): label},
                        label_pos=0.5,
                        font_size=6,
                        font_color="k",
                        font_family="sans-serif",
                        font_weight="normal",
                        alpha=alpha,
                        bbox=dict(alpha=0),
                        horizontalalignment="center",
                        verticalalignment="center",
                        ax=None,
                        rotate=True,
                    )
    _ = plt.tight_layout()
    if save is not None:
        _ = plt.savefig(save, dpi=400)
        print(f"saved as {save}")
    _ = plt.show()

    if returnMainPath:
        transition_pairs = [[k] for k in sidechain.keys()]
        transition_pairs_tmp = [
            [transition_pairs[i], transition_pairs[i + 1]]
            for i in range(len(transition_pairs) - 1)
        ]
        transition_pairs = transition_pairs_tmp + [
            [transition_pairs[-1], transition_pairs[0]]
        ]
        if returnCycleProb:
            return cycles_dict, transition_pairs
        else:
            return transition_pairs
    if returnCycleProb:
        return cycles_dict


def _compute_first_passage_times(
        occupancy, jumps, initial_states,
        final_states, n_jump_per_cycle=5, backward=False):
    """compute hitting time for transition pairs within one permeation event,
        i.e. abs(k_netjumps) < n_bs_jump+1

    Parameters
    ----------
    occupancy: array of size N
        tranjectory expressed in the form of SF occupancy

    jumps: arrays of size (N-1, 2)
        net jumps for ion and water for one trajectory

    initial_states: list of strings
        the SF occupation state that the transitions start in

    final_states: list of strings
        the SF occupation state that the transitions end in

    n_jump_per_cycle: int
        # of ions that constitute a complete permeation cycle.
        Example:
            n_jump_per_cycle = 5 if only jumps from or to S1, S2, S3, and S4
            are considered.

    backward: boolean, False by default
        whether the hitting times corresponding to transitions in which the
        ion movement is against the gradient. If True, then only
        transitions with +ve k_netjump are taken into account

    Returns
    -------
    fpts: list of int
        First passage time

    k_netjumps_count: list of int
        # net k jumps involved in the transitions

    w_netjumps_count: list of int
        # net water jumps involved in the transitions
    """

    # initial_states should contain 1 state only
    fpts = []

    # count number of k and w jumps happening during the period for which
    # hitting times are computed
    k_netjumps_counts = []
    w_netjumps_counts = []

    k_netjumps = jumps[:, 0]
    w_netjumps = jumps[:, 1]

    waiting = False
    passage_time = 0

    # count for each initial state to avoid overlooking some of the initial
    # states in scenario like
    #    initialState_1 -> xxx -> initialState_2 -> xxx -> finalState_2
    # where initialState_2 won't be counted if otherwise
    for int_states in initial_states:
        for i, s in enumerate(occupancy):
            if s in int_states and waiting is False:
                waiting = True
                start_idx = i
            elif s in final_states and waiting is True:
                end_idx = i
                if jumps is None:
                    k_netjump = 0
                else:
                    k_netjump = int(np.sum(k_netjumps[start_idx:end_idx]))
                    w_netjump = int(np.sum(w_netjumps[start_idx:end_idx]))

                # restrict the scope to transitions within one permeation
                # event, i.e. 0 <= abs(k_netjump) < n_jump_per_cycle
                if (
                    (
                        k_netjump >= 0
                        and k_netjump < n_jump_per_cycle
                        and backward is False
                    )
                    or (
                        k_netjump <= 0
                        and k_netjump > -n_jump_per_cycle
                        and backward is True
                    )
                    or n_jump_per_cycle == 0
                ):
                    passage_time = end_idx - start_idx
                    fpts.append(passage_time)
                    k_netjumps_counts.append(k_netjump)
                    w_netjumps_counts.append(w_netjump)

                waiting = False
                start_idx = None
                end_idx = None

    return fpts, k_netjumps_counts, w_netjumps_counts


def _compute_mfpt(
    occupancy_all,
    jumps_all,
    pairs,
    n_jump_per_cycle=5,
    dt=0.02,
    backward=False,
    batch=10000,
    n_resamples=10000,
):
    """compute mean first passage times (MFPTs) for all transition pairs

    Parameters
    ----------
    occupancy_all: list of arrays of size N
        all tranjectories expressed in the form of SF occupancy

    jumps_all: list of arrays of size (N-1, 2)
        net jumps for ion and water for all trajectories

    pairs: list of lists of lists of strings
        pairs[i][0] is the list containing all strings of initial states
        in the i-th transition pair
        pairs[i][1] is the list containing all strings of final states
        in the i-th transition pair

    n_jump_per_cycle: int
        # of ions that constitute a complete permeation cycle.
        Example:
            n_jump_per_cycle = 5 if only jumps from or to S1, S2, S3, and S4
            are considered.

    dt: float
        lag time in ns

    Returns
    -------
    df: Pandas dataframe
        all computed mean first passage times
    """
    data = []
    hts_output = {}
    for initial_states, final_states in pairs:
        fpts_all = []
        k_j_counts_all = []
        w_j_counts_all = []
        initial_states_label = ",".join(initial_states)
        final_states_label = ",".join(final_states)

        for occupancy, jumps in zip(occupancy_all, jumps_all):
            fpts, k_j_counts, w_j_counts = _compute_first_passage_times(
                occupancy,
                jumps,
                initial_states,
                final_states,
                n_jump_per_cycle=n_jump_per_cycle,
                backward=backward,
            )
            fpts_all += fpts
            k_j_counts_all += k_j_counts
            w_j_counts_all += w_j_counts

        fpts_all = np.asarray(fpts_all) * dt
        n_fpts = len(fpts_all)

        if n_fpts > 1:
            fpts_all_mean = np.mean(fpts_all)

            if np.std(fpts_all) > 0.0:
                fpts_all_bs = scipy.stats.bootstrap(
                    (fpts_all,),
                    np.mean,
                    confidence_level=0.95,
                    batch=batch,
                    n_resamples=n_resamples,
                    method="BCa",
                )
                fpts_all_bs_l, fpts_all_bs_u = fpts_all_bs.confidence_interval
            else:
                fpts_all_bs_l, fpts_all_bs_u = 0.0, 0.0

            k_j_counts_all_mean = np.mean(k_j_counts_all)
            w_j_counts_all_mean = np.mean(w_j_counts_all)
        else:
            fpts_all_mean = 0.0
            fpts_all_bs_l, fpts_all_bs_u = 0.0, 0.0
            k_j_counts_all_mean, w_j_counts_all_mean = 0.0, 0.0

        row = [
            initial_states_label,
            final_states_label,
            fpts_all_mean,
            fpts_all_bs_l,
            fpts_all_bs_u,
            n_fpts,
            k_j_counts_all_mean,
            w_j_counts_all_mean,
        ]

        hts_output[initial_states_label + "-" +
                   final_states_label] = np.array(fpts_all)
        data.append(row)

    df = pd.DataFrame(
        data,
        columns=[
            "initial",
            "final",
            "mean (ns)",
            "low (ns)",
            "high (ns)",
            "n",
            "k_f",
            "w_f",
        ],
    )
    return df, hts_output


def _plot_netflux(occupancy_all, weight_threshold=0.1,
                  save=None, returnGraphData=False):
    """plot net fluxes

    Parameters
    ----------
    occupancy_all: list of arrays of size N
        all tranjectories expressed in the form of SF occupancy

    weight_threshold: float
        edges with weight less than 0.2 * maximium of the edge weight are not
        shown

    save: str
        if save is not None, then the plot will be saved to the location
        specified in save.

    Returns
    -------

    edges_weights_full_positive: dict
        key: tuple of str
            (source state, target state)
            value: float
                unnormalized net flux from source state to target state

    states_probs_full: dict
        key: str
            state

        value: float
            steady-state distribution between 0 and 1

    """

    node_size_multiplier = 1000

    prob_dict, prob_matrix, s2i, i2s = _compute_trans_prob(
        occupancy_all, return_matrix=True, quiet=True
    )
    n_states = len(prob_matrix)
    state_probs = np.array([prob_dict[i2s[i]]["prob"]
                           for i in range(n_states)])
    netflux = (
        np.diag(state_probs) @ prob_matrix -
        (np.diag(state_probs) @ prob_matrix).T
    )

    edges_weights_full = {}
    for s1, w1 in s2i.items():
        for s2, w2 in s2i.items():
            edges_weights_full[(s1, s2)] = netflux[w1, w2]

    edges_weights_full_positive = {
        (s1, s2): w for (s1, s2), w in edges_weights_full.items() if w > 0.0
    }

    states_probs_full = {i2s[i]: prob_dict[i2s[i]]["prob"]
                         for i in range(n_states)}

    DG = nx.DiGraph()

    # edges normalized by the maximal net flux
    # and edges with weight < threshold are discarded
    max_weight = max(edges_weights_full_positive.values())
    edges_weights_trunc_norm = [
        (s1, s2, w / max_weight)
        for ((s1, s2), w) in edges_weights_full_positive.items()
        if w / max_weight > weight_threshold
    ]
    DG.add_weighted_edges_from(edges_weights_trunc_norm)

    # nodes, size scales and is normalized by steady-state distribution
    pos = nx.circular_layout(DG)
    node_size = np.array([states_probs_full[s] for s in DG.nodes])
    node_size = node_size / max(node_size) * node_size_multiplier

    # draw
    _ = nx.draw_networkx_nodes(
        DG, pos, node_color="orange",
        linewidths=0, node_size=node_size, alpha=0.5
    )
    _ = nx.draw_networkx_labels(DG, pos)
    _ = nx.draw_networkx_edges(
        DG,
        pos,
        min_source_margin=20,
        min_target_margin=20,
        edgelist=list(edges_weights_trunc_norm),
        alpha=[e[-1] for e in edges_weights_trunc_norm],
    )

    if save is not None:
        _ = plt.savefig(save, dpi=300)
        print(f"saved as {save}")
    _ = plt.show()

    if returnGraphData:
        return states_probs_full, edges_weights_full_positive
