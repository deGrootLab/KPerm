import re
import os
import numpy as np
import scipy.stats
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

class Channel_old:
    def __init__(self):
        self.occupancy_4_all = []
        self.occupancy_6_all = []
        self.jumps_all = []
        self.jumps_4_all = []
        self.jumps_6_all = []
        self.dts = None
        self.dt = None
        self.currents = None
        self.current = None
        self.total_times = None

def loadResults(results_loc):
    """
    Parameters
    ----------
    results_loc: list of strings
        directories in which the results (results.csv and results.log) are stored

    Returns
    -------
    channel: object
        info about the channel, defined by the class Channel

    """
    jumps_all = []
    occupancy_4_all = []
    occupancy_6_all = []
    total_times = []
    dts = []
    currents = []


    for loc in results_loc:
        data_loc = os.path.join(loc, 'results.csv')
        df = pd.read_csv(data_loc, index_col=0)
        occ = df['occupancy'].to_numpy().astype(str)
        jumps = df[['j_k', 'j_w']].to_numpy().astype(int)

        occupancy_6_all.append(occ)
        occ_4 = np.array([s[1:-1] for s in occ])
        occupancy_4_all.append(occ_4)
        jumps_all.append(jumps)

        log_loc = os.path.join(loc, 'results.log')
        with open(log_loc, 'r') as f:
            log = f.read()
        total_time = float(re.search(r'Total time\D+(\d+\.\d+)', log).group(1))
        dt = float(re.search(r'dt\D+(\d+\.\d+)', log).group(1))
        current = float(re.search(r'Current\D+(\d+\.\d+)', log).group(1))
        total_times.append(total_time)
        dts.append(dt)
        currents.append(current)
    channel = Channel_old()
    channel.occupancy_4_all = occupancy_4_all
    channel.occupancy_6_all = occupancy_6_all
    channel.jumps_all = jumps_all
    channel.total_times = np.array(total_times)
    channel.dts = np.array(dts)
    channel.currents = np.array(currents)

    return channel

def computeStats(channel, save=''):
    """
    Parameters
    ----------
    channel: object
        info about the channel, defined by the class Channel
        
    save: string
        save stats to the location specified

    Returns
    -------
    stats: Pandas dataframe
        store statistics of trajectories, such as total time, lag time, current
        and occupation state populations

    states: Pandas dataframe
        point estimate and bootstrap confidence interval for occupation states
    """
    
    try:
        current_bs = scipy.stats.bootstrap((channel.currents,), np.mean, confidence_level=.95, n_resamples=10000, method='BCa')
        current_bs_l, current_bs_h = current_bs.confidence_interval
        channel.current = (np.mean(channel.currents), current_bs_l, current_bs_h)
        print(f"Current (pA): {channel.current[0]:.3f}\t{current_bs_l:.3f} - {current_bs_h:.3f}\n")
    except:
        channel.current = (np.mean(channel.currents), np.mean(channel.currents), np.mean(channel.currents))
        print(f"Current (pA): {channel.current[0]:.3f}\n")


    states, counts = np.unique(np.concatenate(channel.occupancy_6_all), return_counts=True)
    sort_idx = np.argsort(counts)[::-1]
    states = states[sort_idx]
    population = counts[sort_idx] / np.sum(counts)

    stats_dict = {'T (ns)':channel.total_times,
                  'dt (ns)':channel.dts,
                  'current (pA)':channel.currents}

    states_dict = {}
    states_dict['state'] = states
    states_dict['p_mean'] = population

    p_ls = []
    p_hs = []

    for s, p_mean in zip(states, population):
        ps = np.array([np.mean(occupancy == s) for occupancy in channel.occupancy_6_all])
        stats_dict[s] = ps
        try:
            p_bs = scipy.stats.bootstrap((ps,), np.mean, confidence_level=.95, n_resamples=10000, method='BCa')
            p_l, p_h = p_bs.confidence_interval
            p_ls.append(p_l)
            p_hs.append(p_h)
        except:
            p_ls.append(p_mean)
            p_hs.append(p_mean)

    states_dict['p_l'] = p_ls
    states_dict['p_h'] = p_hs

    stats = pd.DataFrame(stats_dict)
    states = pd.DataFrame(states_dict)
    
    if save:
        save = os.path.abspath(save)
        stats.to_csv(save)
        print(f"Stats saved to {save}")
    
    return stats, states

def permeationEventsPartition(occupancy, jump, seedState, n_bs_jump):
    """ partitioning trajectory into permeation events

    Parameters
    ----------
    occupancy: array of size N
        tranjectory expressed in the form of SF occupancy

    jumps: arrays of size (N-1, 2)
        net jumps for ion and water for all trajectories

    seedState: string
        the SF occupation state that the cycles start and end in

    n_bs_jump: int
        # binding sites considered in permeation
        It is used to compute number of k jumps (n_bs_jump+1) one complete
        permeation event takes

    Returns
    -------
    stationaryPhase_indices: list of tuples
        index for the start and the end of stationary phases
    permeationCycle_indices: array of int
        index for the start and the end of permeation cycles, which are
        stationary phase (accumulated jump remains as  int((n_bs_jump+1)*i+offset)
        + conduction phase (accumulated jump increases from
        int((n_bs_jump+1)*i+offset to int((n_bs_jump+1)*(i+1)+offset)
    """
    k_jump = jump[:, 0]
    w_jump = jump[:, 1]

    n_k_netjumps = np.sum(jump[:, 0]) // (n_bs_jump+1)

    # prepend the cumsum with a "0" to align it with occ,
    # now k_netjump_cum[i] is defined as the accumulated
    # k net jump before t=i
    k_netjump_cum = np.zeros(len(k_jump)+1, dtype=int)
    k_netjump_cum[1:] = np.cumsum(k_jump)

    # ignore the last state as no jump info is available
    # and find "seed" with no k and w jump
    indices = np.argwhere(occupancy[:-1] == seedState).reshape(-1)
    # 2. finding "seed" with no k and w jump
    indices = indices[~np.any(jump[indices], axis=1)]
    seed_idx = indices[0]
    offset = k_netjump_cum[seed_idx]

    stationaryPhase_indices = []
    for i in range(n_k_netjumps):
        indices_i = indices[k_netjump_cum[indices] == int((n_bs_jump+1)*i+offset)]
        if len(indices_i) > 0:
            start_i, end_i = indices_i[0], indices_i[-1]
            stationaryPhase_indices.append((start_i, end_i))
        else:
            stationaryPhase_indices.append(())

    # permeation cycle = stationary phase + conduction phase
    #                  = start of i-th SP to start of (i+1)-th SP
    permeationCycle_indices = []
    for i in range(len(stationaryPhase_indices)-1):
        try:
            start_i = stationaryPhase_indices[i][0]
            end_i = stationaryPhase_indices[i+1][0]
            permeationCycle_indices.append([start_i, end_i])
        except:
            print(f"{i}-th cycle is discarded as {seedState} is not found")
            continue

    #print(f"Total permeation events: {n_k_netjumps}")
    #print(f"Identified cycles: {len(permeationCycle_indices)}")
    permeationCycle_indices = np.array(permeationCycle_indices)

    return stationaryPhase_indices, permeationCycle_indices


def permEventsPartition(occupancy, k_jump, cycle_state, n_bs_jump):
    """ partitioning trajectory into full permeation cycles

    Parameters
    ----------
    occupancy: array of size (N, )
        tranjectory expressed in the form of SF occupancy

    k_jump: arrays of size (N, )
        net jumps for ions in the associated cycle_original

    cycle_state: string
        the SF occupation state that the cycles start and end in

    n_bs_jump: int
        # binding sites considered in permeation
        It is used to compute number of k jumps (n_bs_jump+1) one complete
        permeation event takes

    Returns
    -------
    perm_indices: array of int of size (N, 2)
        index for the start and the end of permeation cycles
    """
        
    L = len(occupancy)
    
    t = 0
    T = 0
    found = False
    
    perm_indices = []
    
    while t < L-1:
        if occupancy[t] == cycle_state:
            T = 0
            found = False
            
            while T < L-t-1 and found == False:
                T = T + 1
                
                if occupancy[t+T] == cycle_state:
                    n_k_jump = np.sum(k_jump[t:t+T])
                    
                    if n_k_jump == 5:
                        perm_index_pair = [t, t+T]
                        perm_indices.append(perm_index_pair)
                        found = True
                    elif n_k_jump <= -5 or n_k_jump >= 10:
                        found = True
                        print(f"Not returning to {cycle_state}, no cycle is formed")
            t = t + T - 1
        t = t + 1
        
    perm_indices = np.array(perm_indices)
    
    return perm_indices

def cycleCompression_old(occupancy_cycle, k_jumps_sub, n_bs_jump):
    """ Given an uncompressed (involving osciliation between states without net jumps)
        trajectory segment that starts and ends in the same state and records one
        complete permeation event + the associated jump vectors, compute the "cleaned"
        cycle keeping only the first hit states

    Parameters
    ----------
    occupancy_cycle: array of size (N, )
        occupancy for a trajectory segment

    k_jumps_sub: array of size (N, )
        net jumps for ion in the associated occupancy_cycle

    n_bs_jump: int
        Number of binding sites considered in permeation
        #(n_bs_jump+1) jumps make one complete permeation cycle

    Returns
    -------
    occupancy_compressed: array of size (M, )
        "cleaned" cycle keeping only the first hit states
    """
    _, idx = np.unique(occupancy_cycle, return_index=True)
    unique = occupancy_cycle[np.sort(idx)]

    for state in unique:
        keep = np.ones(len(occupancy_cycle), dtype=bool)
        indices = np.argwhere(occupancy_cycle == state).reshape(-1)

        for i in range(len(indices)-1):
            start_i = indices[i]
            end_i = indices[i+1]
            k_netjumps_i = k_jumps_sub[start_i:end_i]

            # discard if the state repeats itself without net ion jump
            if (end_i - start_i) > 0 and np.sum(k_netjumps_i) == 0:
                keep[start_i:end_i] = False

        occupancy_cycle = occupancy_cycle[keep]
        k_jumps_sub = k_jumps_sub[keep]

    occupancy_compressed = occupancy_cycle
    k_jumps_compressed = k_jumps_sub

    if np.sum(k_jumps_compressed) != (n_bs_jump+1):
        print(f"netjump != {(n_bs_jump+1)}")
    return occupancy_compressed

def cycleReduction(cycle_original, k_jumps_sub, n_bs_jump):
    """ Given an original (involving osciliation between states without net jumps)
        trajectory segment (cycle) that starts and ends in the same state and records one
        complete permeation event, and the associated jump vectors, compute the "cleaned"
        cycle keeping only the first-hit states

    Parameters
    ----------
    cycle_original: array of size (N, )
        Original cycle of SF occupancy representing one permeation event. 
        The first and the last state has to be the same.

    k_jumps_sub: array of size (N, )
        net jumps for ion in the associated cycle_original

    n_bs_jump: int
        Number of binding sites considered in permeation
        #(n_bs_jump+1) jumps make one complete permeation cycle

    Returns
    -------
    occupancy_compressed: array of size (M, )
        "cleaned" cycle keeping only the first hit states
    """
    assert cycle_original[0] == cycle_original[-1], "First and last state not the same"
    
    T = len(cycle_original)
    
    t = 0
    cycle_reduced = []
    
    while t < T-1:
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

def findCycles(occupancy_all, jumps_all, seedState, n_bs_jump=4):
    """ Given occupancy and jumps of the trajectories, the seed state, and n_bs_jump
    that define the # BSs in which jumps in and out are considered,
    give the cycles that start and end in the seed state

    Parameters
    ----------
    occupancy_all: list of arrays of size N
        occupancy for all trajectories

    jumps_all: list of arrays of size (N-1, 2)
        net jumps for ion and water for all trajectories

    seedState: string
        the SF occupation state that the cycles start and end in

    n_bs_jump: int
        # binding sites considered in permeation
        It is used to compute number of k jumps (n_bs_jump+1) one complete
        permeation event takes

    Returns
    -------
    permeationCycle_indices: list of lists of arrays
        contain the cycles identified in each trajectory
    """
    permeationCycles = []
    p_indices_all = []

    n_k_netjumps = []
    n_identified_cycles = []

    for i, (occupancy, jumps) in enumerate(zip(occupancy_all, jumps_all)):
        print(f"Trajectory {i}")
        # _ , p_indices = permeationEventsPartition(occupancy, jumps, seedState, n_bs_jump=n_bs_jump)
        # p_indices_all.append(p_indices)

        p_indices = permEventsPartition(occupancy, jumps, seedState, n_bs_jump=n_bs_jump)
        p_indices_all.append(p_indices)
        

        # permeationCycles_ = [cycleCompression(occupancy[i:j+1], jumps[i:j+1,0], n_bs_jump=n_bs_jump)
        #                      for (i, j) in p_indices]
        permeationCycles_ = [cycleReduction(occupancy[i:j+1], jumps[i:j+1,0], n_bs_jump=n_bs_jump)
                             for (i, j) in p_indices]
        permeationCycles.append(permeationCycles_)

        n_k_netjump = np.sum(jumps[:, 0]) // (n_bs_jump+1)
        n_identified_cycle = len(p_indices)

        n_k_netjumps.append(n_k_netjump)
        n_identified_cycles.append(n_identified_cycle)

        print(f"Number of permeation events: {int(n_k_netjump)}")
        print(f"Number of identified cycles: {n_identified_cycle} \t {n_identified_cycle/n_k_netjump * 100:.2f}%\n")


    
    print(f"Total number of permeation events: {int(np.sum(n_k_netjumps))}")
    print(f"Total number of identified cycles: {np.sum(n_identified_cycles)} \t {np.sum(n_identified_cycles)/np.sum(n_k_netjumps) * 100:.3f}%\n\n")


    return permeationCycles, p_indices_all

def computeTransProb(trajs, return_matrix=False, quiet=False):
    """ given trajectories, compute transition probabilities

    Parameters
    ----------
    trajs: list of arrays of size N    or   list of lists of arrays
        trajectories, can be occupancy or cycles for all trajectories

    return_matrix: boolean
        if True, return also the transition matrix and mapping between indices and state names

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

#     trans_prob_dict = {state:{"count":count, "prob":count/counts_total} \
#                        for state, count in zip(states, counts)}
    trans_prob_dict = {state:{} for state in states}

    state_to_idx = {s:i for i, s in enumerate(states)}
    idx_to_state = {i:s for i, s in enumerate(states)}

    n_states = len(states)
    trans_counts = np.zeros((n_states, n_states), dtype=int)

    for traj in trajs:
        for t in range(len(traj)-1):
            i, j = state_to_idx[traj[t]], state_to_idx[traj[t+1]]
            trans_counts[i,j] += 1
    # trans_counts[trans_counts < threshold] = 0
    trans_prob = np.nan_to_num(np.asarray([trans_counts[i] / (np.sum(trans_counts[i]) or 1.0) for i in range(n_states)]))

    n_outward_total = np.sum(trans_counts, axis=1)

    for i in range(n_states):
        state_i = idx_to_state[i]
        n_total = n_outward_total[i]


        indices = np.argsort(trans_prob[i])[::-1]
        for j in indices:
            state_j = idx_to_state[j]
            p = trans_prob[i, j]
            if n_total < 2:
                err = .0
            else:
                err = 1.96 * np.sqrt(p*(1-p)/(n_total-1))
            n = trans_counts[i, j]
            trans_prob_dict[state_i][state_j] = (trans_prob[i, j], err, n)

        trans_prob_dict[state_i]['count_out'] = n_total
        trans_prob_dict[state_i]['count'] = counts[i]
        trans_prob_dict[state_i]['prob'] = counts[i]/counts_total

    if not quiet:
        for state_i, state_i_dict in trans_prob_dict.items():
            print(f"\n========= {state_i} {state_i_dict['count_out']}===========")
            for k, v in state_i_dict.items():
                if k not in ['count', 'prob', 'count_out']:
                    state_j = k
                    p, err, n = v
                    if p != .0:
                        print(f"{state_j:<10}\t{p:<12.5%} ± {err:<12.5%} {n}")

    if return_matrix:
        return trans_prob, state_to_idx, idx_to_state
    else:
        return trans_prob_dict

def plotCycles(cycles, state_threshold=.05, label_threshold=.05, offset=0.1, scale=0.1,
               figsize=(10,10), save=None, returnCycleProb=False, returnMainPath=False):
    """ given trajectories, compute transition probabilities

    Parameters
    ----------
    cycles: list of lists of arrays
        contain the cycles identified in each trajectory

    Returns
    -------
    returnCycleProb: boolean
        If True, return cycles_dict
            cycles_dict[state_i][state_j] is the probability of observing the transition
            state_i -> state_j in a permeation cycle which starts and ends in a specified state
            *** so these probs are seed state-dependent ***

    returnMainPath: boolean
        If True, return the dominant cycle for determining MFPT using permeationMFPT()

    """
    # flatten nested cycles
    cycles_flattened = [c for cycle in cycles for c in cycle]
    n_cycles = len(cycles_flattened)

    # probs: dict, key=state, value=(target state, %, count)
    cycles_dict = computeTransProb(cycles_flattened, quiet=True)

    ############################################################

    # find the backbone of the cyclic graph
    states_all = np.array([k for k in cycles_dict.keys()])
    state_counts = np.array([cycles_dict[state]['count_out'] for state in states_all])
    state_counts_total = np.sum(state_counts)
    state_p = state_counts / state_counts_total
    states_selected = states_all[state_p > state_threshold]

    # assume that states are already sorted in descending order of population
    seedState = list(cycles_dict)[0]

    backbone = []
    backbone.append(seedState)

    state = list(cycles_dict[seedState].keys())[0]
    backbone.append(state)

    while(backbone[-1] != seedState):
        state = list(cycles_dict[state].keys())[0]
        backbone.append(state)

    # assign the side branches of the cyclic graph
    non_backbone = states_selected[np.in1d(states_selected, backbone, invert=True)]
    sidechain = {k:[] for k in backbone}

    # put next to the backbone state before the target state which the non-backbone
    # state has highest probability transitioning to
    for state_i in non_backbone:
        j_idx = None
        for state_j in list(cycles_dict[state_i]):
            try:
                j_idx = backbone.index(state_j)
                break
            except:
                continue
        sidechain[backbone[j_idx-1]].append(state_i)

    ############################################################
    # determine position of nodes

    # centered at origin by default
    pos = nx.circular_layout(nx.path_graph(backbone))
    for state_backbone, states_sidechain in sidechain.items():
        vec = pos[state_backbone] / np.linalg.norm(pos[state_backbone])
        for j, state_sidechain in enumerate(states_sidechain):
            pos[state_sidechain] = pos[state_backbone] + offset * vec + vec * j * scale

    ############################################################
    # initialize graph
    _ = plt.figure(1,figsize=figsize, dpi=300)
    _ = plt.axis('equal')
    G = nx.DiGraph()
    G.add_nodes_from(states_selected)
    sizes = np.array([cycles_dict[state]['count_out'] for state in states_selected])

    nx.draw_networkx(G, pos, node_color='orange',
                     node_size=100*sizes/n_cycles,
                     alpha=1, font_size=6, font_color='k')
    ############################################################

    for state_i in states_all:
        _ = cycles_dict[state_i].pop('count')
        cycles_dict[state_i]['prob'] = cycles_dict[state_i]['count_out'] / n_cycles
        for state_j in states_all:
            # here p is defined as the probability of having this transition in every
            # observed permeation cycle, different from the Markov transition prob
            _, _, count = cycles_dict[state_i][state_j]
            p = count / n_cycles
            if n_cycles < 2:
                err = 0
            else:
                err = 1.96 * np.sqrt(p*(1-p)/ (n_cycles-1))
            cycles_dict[state_i][state_j] = (p, err, count)

    for state_i in states_selected:
        for state_j in states_selected:
            if state_i != state_j:
                p, err, _ = cycles_dict[state_i][state_j]
                alpha = np.tanh(3*p)/np.tanh(3)
                _ = nx.draw_networkx_edges(G, pos,
                                           edgelist=[(state_i, state_j, 2)],
                                           arrowsize=12,
                                           min_source_margin=10, min_target_margin=10,
                                           alpha=alpha,
                                           connectionstyle="arc3,rad=0.2")
                if p > label_threshold:
                    label = fr"{p*100:.1f}$\pm${err*100:.1f}%"
                    _ = nx.draw_networkx_edge_labels(G, pos, edge_labels={(state_i, state_j):label} ,
                                                     label_pos=0.5, font_size=6,
                                                     font_color='k', font_family='sans-serif',
                                                     font_weight='normal', alpha=alpha, bbox=dict(alpha=0),
                                                     horizontalalignment='center', verticalalignment='center',
                                                     ax=None, rotate=True)
    _ = plt.tight_layout()
    if save is not None:
        _ = plt.savefig(save, dpi=400)
        print(f"saved as {save}")
    _ = plt.show()

    if returnMainPath:
        transition_pairs = [[k]for k in sidechain.keys()]
        transition_pairs_tmp = [[transition_pairs[i], transition_pairs[i+1]] for i in range(len(transition_pairs)-1)]
        transition_pairs = transition_pairs_tmp + [[transition_pairs[-1], transition_pairs[0]]]
        if returnCycleProb:
            return cycles_dict, transition_pairs
        else:
            return transition_pairs
    if returnCycleProb:
        return cycles_dict

def hittingTimes(occupancy, jumps, intStates, finalStates, n_bs_jump=4, backward=False):
    """ compute hitting time for transition pairs within one permeation event,
        i.e. abs(k_netjumps) < n_bs_jump+1

    Parameters
    ----------
    occupancy: array of size N
        tranjectory expressed in the form of SF occupancy

    jumps: arrays of size (N-1, 2)
        net jumps for ion and water for one trajectory

    intStates: list of strings
        the SF occupation state that the transitions start in

    finalStates: list of strings
        the SF occupation state that the transitions end in

    n_bs_jump: int
        # binding sites considered in permeation
        It is used to compute number of k jumps (n_bs_jump+1) one complete
        permeation event takes

    backward: boolean, False by default
        whether the hitting times corresponding to transitions in which the ion movement is
        against the gradient. If True, then only transitions with +ve k_netjump are taken into account

    Returns
    -------
    hittingTimes: list of int

    k_netjumps_count: list of int
        # net k jumps involved in the transitions

    w_netjumps_count: list of int
        # net water jumps involved in the transitions
    """

    #intStates should contain 1 state only
    hittingTimes = []

    # count number of k and w jumps happening during the period for which hitting times are computed
    k_netjumps_counts = []
    w_netjumps_counts = []

    k_netjumps = jumps[:, 0]
    w_netjumps = jumps[:, 1]

    waiting = False
    hittingTime = 0

    # count for each initial state to avoid overlooking some of the initial states in scenario like
    #    initialState_1 -> xxx -> initialState_2 -> xxx -> finalState_2
    # where initialState_2 won't be counted if otherwise
    for intState in intStates:
        for i, s in enumerate(occupancy):
            if s in intState and waiting is False:
                waiting = True
                start_idx = i
            elif s in finalStates and waiting is True:
                end_idx = i
                if jumps is None:
                    k_netjump = 0
                else:
                    k_netjump = int(np.sum(k_netjumps[start_idx:end_idx]))
                    w_netjump = int(np.sum(w_netjumps[start_idx:end_idx]))

                # restrict the scope to transitions within one permeation event,
                # i.e. 0 <= abs(k_netjump) < (n_bs_jump+1)
                if (k_netjump >= 0 and k_netjump < (n_bs_jump+1) and backward is False) or \
                    (k_netjump <= 0 and k_netjump > -(n_bs_jump+1) and backward is True) or n_bs_jump == 0:
                    #print(n_bs_jump)

                    hittingTime = end_idx - start_idx
                    hittingTimes.append(hittingTime)
                    k_netjumps_counts.append(k_netjump)
                    w_netjumps_counts.append(w_netjump)

                waiting = False
                start_idx = None
                end_idx = None

    return hittingTimes, k_netjumps_counts, w_netjumps_counts

def permeationMFPT(occupancy_all, jumps_all, pairs, n_bs_jump=4, dt=.02, backward=False, batch=10000, n_resamples=10000):
    """ compute hitting time for all transition pairs

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

    n_bs_jump: int
        # binding sites considered in permeation
        It is used to compute number of k jumps (n_bs_jump+1) one complete
        permeation event takes

    dt: float
        lag time in ns

    Returns
    -------
    df: Pandas dataframe
        all computed mean first passage times
    """
    data = []
    hts_output = {}
    for initialStates, finalStates in pairs:
        hTs_all = []
        k_j_counts_all = []
        w_j_counts_all = []
        inititalStates_label = ','.join(initialStates)
        finalStates_label = ','.join(finalStates)

        for occupancy, jumps in zip(occupancy_all, jumps_all):
            hTs, k_j_counts, w_j_counts = hittingTimes(occupancy, jumps, initialStates, finalStates,
                                                                  n_bs_jump=n_bs_jump, backward=backward)
            hTs_all += hTs
            k_j_counts_all += k_j_counts
            w_j_counts_all += w_j_counts

        hTs_all = np.asarray(hTs_all) * dt
        n_hTs = len(hTs_all)

        if n_hTs > 1:
            hTs_all_mean = np.mean(hTs_all)

            if np.std(hTs_all) > .0:
                hTs_all_bs = scipy.stats.bootstrap((hTs_all, ), np.mean, confidence_level=.95, batch=batch,
                                                   n_resamples=n_resamples, method='BCa')
                hTs_all_bs_l, hTs_all_bs_u = hTs_all_bs.confidence_interval
            else:
                hTs_all_bs_l, hTs_all_bs_u = .0, .0

            k_j_counts_all_mean = np.mean(k_j_counts_all)
            w_j_counts_all_mean = np.mean(w_j_counts_all)
        else:
            hTs_all_mean = .0
            hTs_all_bs_l, hTs_all_bs_u = .0, .0
            k_j_counts_all_mean, w_j_counts_all_mean = .0, .0

        row = [inititalStates_label, finalStates_label, hTs_all_mean,
               hTs_all_bs_l, hTs_all_bs_u, n_hTs, k_j_counts_all_mean, w_j_counts_all_mean]

        hts_output[inititalStates_label+'-'+finalStates_label] = np.array(hTs_all)
        data.append(row)

    df = pd.DataFrame(data,
                      columns=["initial","final", "mean (ns)", "low (ns)",
                               "high (ns)", "n", "k_f", "w_f"])
    return df, hts_output
