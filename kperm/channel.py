"""Channel class and core workflow of KPERM

The module contains the class Channel and core functions for analysizing
ion permeation events in MD simulations using the KPERM framework.

"""
import os
import re
import warnings

import numpy as np
import MDAnalysis as mda
import pandas as pd
from scipy.stats import bootstrap

from kperm.permeation import _find_cycles, _compute_trans_prob, _plot_cycle, \
    _compute_mfpt, _plot_netflux, _count_perm_cross
from kperm.utils import _write_list_of_tuples, _count_time, _create_logger

warnings.simplefilter('always', DeprecationWarning)


class Channel:
    def __init__(self, coor=None, trajs=None):
        self.coor = coor
        self.trajs = trajs
        self.results_loc = []
        self.occupancy_4_all = []
        self.occupancy_6_all = []
        self.jumps_all = []
        self.cycles_all = []
        self.n_water_events = []
        self.n_k_events = []
        self.dts = None
        self.dt = None
        self.currents = None
        self.current = None
        self.total_times = None
        self.stats = None
        self.permeation_idx_all = None
        self.cycle_prob = None
        self.main_path = None

    def run(self, perm_count=("cross"), output="kperm",
            perm_details=False, sf_diag=False):
        self.results_loc = []

        for traj in self.trajs:
            run(
                self.coor,
                traj,
                perm_count=perm_count,
                perm_details=perm_details,
                output=output,
                sf_diag=sf_diag,
            )

            self.results_loc.append(os.path.dirname(traj))

    def load(self, results_loc=None, filename="kperm"):
        jumps_all = []
        occupancy_4_all = []
        occupancy_6_all = []
        total_times = []
        dts = []
        currents = []
        n_water_events = []
        n_k_events = []

        if results_loc is not None:
            self.results_loc = results_loc

        for result_loc in self.results_loc:
            print(f"loading results from {result_loc}/{filename} ...",
                  end=" ")
            result_path = os.path.abspath(
                os.path.join(result_loc, filename + ".csv"))
            log_path = os.path.abspath(
                os.path.join(result_loc, filename + ".log"))

            df = pd.read_csv(result_path, index_col=0)
            occ = df["occupancy"].to_numpy().astype(str)
            jumps = df[["j_k", "j_w"]].to_numpy().astype(int)

            occupancy_6_all.append(occ)
            occ_4 = np.array([s[1:-1] for s in occ])
            occupancy_4_all.append(occ_4)
            jumps_all.append(jumps)

            with open(log_path, "r", encoding="utf-8") as f:
                log = f.read()
            total_time = float(
                re.search(r"Total time\D+(\d+\.\d+)", log).group(1))
            dt = float(re.search(r"dt\D+(\d+\.\d+)", log).group(1))

            total_times.append(total_time)
            dts.append(dt)

            try:
                n_k_jump = re.search(
                    r"net ion permeation events \(jump\) = " +
                    r"([-+]?\d+|N/A)",
                    log,
                ).group(1)
                n_w_jump = re.search(
                    r"net water permeation events \(jump\) = " +
                    r"([-+]?\d+|N/A)",
                    log,
                ).group(1)
                k_current_jump = re.search(
                    r"[cC]urrent \(jump\) = ([-+]?\d+\.\d+|N/A)",
                    log
                ).group(1)

                n_k_cross = re.search(
                    r"net ion permeation events \(cross\) = " +
                    r"([-+]?\d+|N/A)",
                    log,
                ).group(1)
                n_w_cross = re.search(
                    r"net water permeation events \(cross\) = " +
                    r"([-+]?\d+|N/A)",
                    log,
                ).group(1)
                k_current_cross = re.search(
                    r"[cC]urrent \(cross\) = ([-+]?\d+\.\d+|N/A)",
                    log
                ).group(1)

            except AttributeError:
                # for backward compatibility
                n_k_jump = re.search(
                    r"Number of net ion permeation events = ([-+]?\d+|N/A)",
                    log
                ).group(1)
                n_w_jump = re.search(
                    r"Number of net water permeation events = ([-+]?\d+|N/A)",
                    log
                ).group(1)
                k_current_jump = re.search(
                    r"[cC]urrent = ([-+]?\d+\.\d+|N/A)", log).group(1)

                n_k_cross = "N/A"
                n_w_cross = "N/A"
                k_current_cross = "N/A"

            if n_k_jump != "N/A":
                n_k_events.append(int(n_k_jump))
                n_water_events.append(int(n_w_jump))
                currents.append(float(k_current_jump))
                print("permeation counts based on jumps are used.")

            elif n_k_cross != "N/A":
                n_k_events.append(int(n_k_cross))
                n_water_events.append(int(n_w_cross))
                currents.append(float(k_current_cross))
                print(
                    "permeation counts based on jumps are not found, counts \
based on cross are used."
                )

            else:
                print("No permeation count data are available.")

        print("Loading finished.")

        self.occupancy_4_all = occupancy_4_all
        self.occupancy_6_all = occupancy_6_all
        self.jumps_all = jumps_all
        self.total_times = np.array(total_times)
        self.dts = np.array(dts)
        self.currents = np.array(currents)
        self.n_water_events = n_water_events
        self.n_k_events = n_k_events

    def compute_stats(self):
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
            store statistics of trajectories, such as total time, lag time,
            current and occupation state populations.

        states: Pandas dataframe
            point estimate and bootstrap confidence interval for
            occupation states.
        """
        print("==============================")

        try:
            current_bs = bootstrap(
                (self.currents,),
                np.mean,
                confidence_level=0.95,
                n_resamples=10000,
                method="BCa",
            )
            current_bs_l, current_bs_h = current_bs.confidence_interval
            self.current = (np.mean(self.currents), current_bs_l, current_bs_h)
            print(
                f"K+ Current (pA): {self.current[0]:.3f}\t" +
                f"{current_bs_l:.3f} - {current_bs_h:.3f}"
                )
        except ValueError:
            self.current = (
                np.mean(self.currents),
                np.mean(self.currents),
                np.mean(self.currents),
            )
            print(f"K+ Current (pA): {self.current[0]:.3f}")

        states, counts = np.unique(
            np.concatenate(self.occupancy_6_all), return_counts=True
        )
        sort_idx = np.argsort(counts)[::-1]
        states = states[sort_idx]
        population = counts[sort_idx] / np.sum(counts)

        stats_dict = {
            "T (ns)": self.total_times,
            "dt (ns)": self.dts,
            "current (pA)": self.currents,
        }

        states_dict = {}
        states_dict["state"] = states
        states_dict["p_mean"] = population

        p_ls = []
        p_us = []

        for s, p_mean in zip(states, population):
            ps = np.array(
                [np.mean(occupancy == s) for occupancy in self.occupancy_6_all]
            )
            stats_dict[s] = ps
            try:
                p_bs = bootstrap(
                    (ps,),
                    np.mean,
                    confidence_level=0.95,
                    n_resamples=10000,
                    method="BCa",
                )
                p_l, p_u = p_bs.confidence_interval
                p_ls.append(p_l)
                p_us.append(p_u)
            except ValueError:
                p_ls.append(p_mean)
                p_us.append(p_mean)

        states_dict["p_l"] = p_ls
        states_dict["p_u"] = p_us
        states = pd.DataFrame(states_dict)
        stats = pd.DataFrame(stats_dict)

        print(f"current (pA): {self.currents}")
        print(f"number of k permeation events: {self.n_k_events}")
        print(f"number of water permeation events: {self.n_water_events}")
        print(f"total time (ns): {self.total_times}")
        print(f"dt (ns): {self.dts}")
        print("coordinate:")
        print(f"\t{self.coor}")
        print("trajectories:")
        if self.trajs is None:
            print(f"\t{self.trajs}")
        else:
            for traj in self.trajs:
                print(f"\t{traj}")
        print("==============================")

        self.stats = stats

    def computeStats(self, *args, **kwargs):
        warnings.warn(
            "computeStats() is renamed to compute_stats(). " +
            "You are recommended to call <Channel>.compute_stats() instead. " +
            "computeStats() will be removed in the future.",
            DeprecationWarning)
        return self.compute_stats(*args, **kwargs)

    def find_cycle(self, cycle_state, n_jump_per_cycle=5):
        if len(cycle_state) == 4:
            cycles, perm_idx_all, cycle_count_perc = _find_cycles(
                self.occupancy_4_all, self.jumps_all, cycle_state,
                n_jump_per_cycle=n_jump_per_cycle
            )

            self.cycles_all = cycles
            self.permeation_idx_all = perm_idx_all
        elif len(cycle_state) == 6:
            cycles, perm_idx_all, cycle_count_perc = _find_cycles(
                self.occupancy_6_all, self.jumps_all, cycle_state,
                n_jump_per_cycle=n_jump_per_cycle
            )

            self.cycles_all = cycles
            self.permeation_idx_all = perm_idx_all
        else:
            raise ValueError("Input invalid.")

        cycles_flattened = [c for cycle in self.cycles_all for c in cycle]
        self.cycle_prob = _compute_trans_prob(cycles_flattened, quiet=True)

        return cycle_count_perc

    def findCycles(self, *args, **kwargs):
        warnings.warn(
            "findCycles() is renamed to find_cycle(). " +
            "You are recommended to call <Channel>.find_cycle() instead. " +
            "findCycles() will be removed in the future.",
            DeprecationWarning)
        return self.find_cycle(*args, **kwargs)

    def plotCycles(self, *args, **kwargs):
        warnings.warn(
            "plotCycles() is renamed to plot_cycle(). " +
            "You are recommended to call <Channel>.plot_cycle() instead. " +
            "plotCycles() will be removed in the future.",
            DeprecationWarning)
        return self.plot_cycle(*args, **kwargs)

    def plot_cycle(
        self,
        state_threshold=0.01,
        label_threshold=0.15,
        offset=0.7,
        scale=0.4,
        figsize=(6, 6)
    ):

        self.cycle_prob, self.main_path = _plot_cycle(
            self.cycles_all,
            state_threshold=state_threshold,
            label_threshold=label_threshold,
            offset=offset,
            scale=scale,
            figsize=figsize,
            cycle_prob=True,
            main_cycle=True,
        )

    def permeationMFPT(self, *args, **kwargs):
        warnings.warn(
            "permeationMFPT() is renamed to mfpt(). " +
            "You are recommended to call <Channel>.mfpt() instead. " +
            "permeationMFPT() will be removed in the future.",
            DeprecationWarning)
        return self.mfpt(*args, **kwargs)

    def mfpt(
        self,
        dt=0.02,
        paths=None,
        n_jump_per_cycle=5,
        backward=False,
        batch=10000,
        n_resamples=10000,
    ):
        if paths is None:
            raise ValueError(
                "Paths are required to compute the mean first passage time."
            )

        if len(paths[0][0][0]) == 4:
            mfpt, fps = _compute_mfpt(
                self.occupancy_4_all,
                self.jumps_all,
                paths,
                n_jump_per_cycle=n_jump_per_cycle,
                dt=dt,
                backward=backward,
                batch=batch,
                n_resamples=n_resamples,
            )
        elif len(paths[0][0][0]) == 6:
            mfpt, fps = _compute_mfpt(
                self.occupancy_6_all,
                self.jumps_all,
                paths,
                n_jump_per_cycle=n_jump_per_cycle,
                dt=dt,
                backward=backward,
                batch=batch,
                n_resamples=n_resamples,
            )
        else:
            raise ValueError("paths invalid.")
        return mfpt, fps

    def plotNetFlux(self, *args, **kwargs):
        warnings.warn(
            "plotNetFlux() is renamed to plot_netflux(). " +
            "You are recommended to call <Channel>.plot_netflux() instead. " +
            "plotNetFlux() will be removed in the future.",
            DeprecationWarning)
        return self.plot_netflux(*args, **kwargs)

    def plot_netflux(self, weight_threshold=0.1, save=None, data=False,
                     returnGraphData=False):
        if returnGraphData is True:
            warnings.warn(
                "returnGraphData is renamed to Data. " +
                "You are recommended to use data=true instead. " +
                "returnGraphData will be removed in the future.",
                DeprecationWarning)

            data = True

        return _plot_netflux(
            self.occupancy_6_all,
            weight_threshold,
            save=save,
            data=data,
        )


def detectSF(*args, **kwargs):
    warnings.warn(
        "detectSF() is renamed to detect_sf(). " +
        "You are recommended to call detect_sf() instead. " +
        "detectSF() will be removed in the future.",
        DeprecationWarning)
    return detect_sf(*args, **kwargs)


def detect_sf(coor, quiet=False, o_cutoff=5, og1_cutoff=7.0, all_res=False):
    """read coordinate file and return zero-indexed atom indices defining SF

    Parameters
    ----------
    coor : str
        The path to the coordinate file containing the ion channel.
    quiet : bool, optional
        A flag used to print detailed information about SF (default is False).
    o_cutoff: float
        Distance cutoff for determining if backbone oxygen atoms are close
        enough to meet the geometric requirement for being a part of SF.
        Unit: Angstrom
    og1_cutoff: float
        Same as o_cutoff, but it is for OG1 atom (like threonine's OG1).
        Unit: Angstrom
    all_res: boolean
        If false, only amino acids GLY VAL TYR THR PHE ILE are considered
        when determining SF.

    Returns
    -------
    sf_layer: dict
        consisting of two dictionaries containing atom idx (zero-based) and
        residue id of SF's oxygen (backbone and hydroxyl) and CA atoms,
        respectively.

    """
    u = mda.Universe(coor, in_memory=False)

    # Finding backbone oxygen atoms that meet the geometric requirements
    if all_res:
        protein_o = u.select_atoms("protein and name O", updating=False)
    else:
        protein_o = u.select_atoms(
            "protein and name O and (resname GLY VAL TYR THR PHE ILE)",
            updating=False,
        )

    resid = np.array([atom.resid for atom in protein_o])
    d_matrix = np.linalg.norm(
        (protein_o.positions[:, np.newaxis] - protein_o.positions), axis=-1
    )
    # not count atoms of themselves and their two nearest neighbours for any
    # of the selected residue type in sequence
    d_matrix = d_matrix + (
        1e5
        * np.tril(np.ones(d_matrix.shape), k=2)
        * np.triu(np.ones(d_matrix.shape), k=-2)
    )
    n_neighbors = np.sum(d_matrix < o_cutoff, axis=1)
    resid = resid[n_neighbors > 2]  # at least 3 neighbors

    tmp = []
    sf_o_resid = []

    # find atoms with consecutive resids
    for i in range(len(resid)):
        if i < len(resid) - 1:
            if resid[i] == resid[i + 1] - 1:
                tmp.append(resid[i])
            elif resid[i] == resid[i - 1] + 1:
                tmp.append(resid[i])
                sf_o_resid.append(tmp)
                tmp = []
        else:
            if resid[i] == resid[i - 1] + 1:
                tmp.append(resid[i])
                sf_o_resid.append(tmp)

    # (CA only) +1 to include the next residue after SF residues
    sf_ca_resid = [resids + [resids[-1] + 1] for resids in sf_o_resid]
    sf_ca_resid = [i for chain in sf_ca_resid[:] for i in chain]
    ###
    sf_o_resid = [resids + [resids[-1] + 1] for resids in sf_o_resid]
    sf_o_resid = [i for chain in sf_o_resid[:] for i in chain]
    sf_o = u.select_atoms(
        f"protein and name O and resid \
{' '.join([str(i) for i in sf_o_resid])}",
        updating=False,
    )
    sf_ca = u.select_atoms(
        f"protein and name CA and resid \
{' '.join([str(i) for i in sf_ca_resid])}",
        updating=False,
    )

    # find OG1 that form binding site(s)
    protein_og1 = u.select_atoms("protein and name OG1", updating=False)
    sf_og1_indices = np.array([atom.ix for atom in protein_og1])
    d_matrix = np.linalg.norm(
        (protein_og1.positions[:, np.newaxis] - protein_og1.positions), axis=-1
    )
    # not count itself and atoms of nearest neighbours having OG1 in sequence
    d_matrix = d_matrix + (
        1e5
        * np.tril(np.ones(d_matrix.shape), k=1)
        * np.triu(np.ones(d_matrix.shape), k=-1)
    )
    n_neighbors = np.sum(d_matrix < og1_cutoff, axis=1)
    sf_og1_indices = sf_og1_indices[n_neighbors > 2]
    sf_og1 = u.select_atoms(
        f"protein and name OG1 and index \
{' '.join([str(i) for i in sf_og1_indices])}",
        updating=False,
    )
    sf_o = u.select_atoms(
        f"protein and index \
{' '.join([str(atom.ix) for atom in sf_o+sf_og1])}",
        updating=False,
    )

    sf_layer = {"O": {}, "CA": {}}
    for sf_atom, name in zip([sf_o, sf_ca], ["O", "CA"]):
        # number of layer = # atoms // 4
        layer_id = len(sf_atom) // 4 - 1
        for atom in sf_atom:
            if layer_id not in sf_layer[name].keys():
                sf_layer[name][layer_id] = {
                    "idx": [atom.ix],
                    "resid": [atom.resid],
                    "resname": [atom.resname],
                    "name": [atom.name],
                }
            else:
                sf_layer[name][layer_id]["idx"].append(atom.ix)
                sf_layer[name][layer_id]["resid"].append(atom.resid)
                sf_layer[name][layer_id]["resname"].append(atom.resname)
                sf_layer[name][layer_id]["name"].append(atom.name)
            layer_id = layer_id - 1 if layer_id != 0 else (len(sf_atom)//4 - 1)

    # rearrange order so that the first atom is the diagonal pair of the second
    # atom, and the third is the diagonal pair of the fourth one
    sf_ca_0_pos = u.select_atoms("all").positions[sf_layer["CA"][0]["idx"]]
    i_oppo_to_0 = np.linalg.norm(sf_ca_0_pos-sf_ca_0_pos[0], axis=1).argmax()

    if not quiet:
        print("idx\tlayer\tchain\tresid\tresname\tname")
    for atomtype in sf_layer:
        for layer in sf_layer[atomtype]:
            for info in sf_layer[atomtype][layer]:
                (
                    sf_layer[atomtype][layer][info][1],
                    sf_layer[atomtype][layer][info][i_oppo_to_0],
                ) = (
                    sf_layer[atomtype][layer][info][i_oppo_to_0],
                    sf_layer[atomtype][layer][info][1],
                )

    if not quiet:
        for layer in range(len(sf_layer["O"].keys())):
            indices = sf_layer["O"][layer]["idx"]
            resids = sf_layer["O"][layer]["resid"]
            resnames = sf_layer["O"][layer]["resname"]
            names = sf_layer["O"][layer]["name"]

            for chain_id, (idx, resid, resname, name) in enumerate(
                zip(indices, resids, resnames, names)
            ):
                print(
                    f"{idx}\t{layer}\t{chain_id}\t{resid}\t{resname}\t{name}"
                )

    return sf_layer


def _get_non_protein_index(coor):
    """read coordinate file and return zero-indexed atom indices defining SF

    Parameters
    ----------
    coor : str
        The path to the coordinate file containing the system

    Returns
    -------
    indices: tuple
        arrays of zero-indexed indices of K, Cl and water respectively

    """
    u = mda.Universe(coor, in_memory=False)

    indices = (
        u.select_atoms("resname K POT", updating=False).ix,
        u.select_atoms("resname CL CLA", updating=False).ix,
        u.select_atoms("(resname SOL TIP3) and (name OH2 OW)",
                       updating=False).ix,
    )

    return indices


def _find_bound(positions, sol_indices, sf_o_indices, additional_bs_cutoff=4.0,
                bs_radius=4.0):
    """read solute (K) or solvent (water) positions and assign their
    (zero-based) indicies to binding sites

    Parameters
    ----------
    coor : str
        The path to the coordinate file containing the system.
    sol_idx : array of int
        containing zero-based indices of particles such as oxygen of water,
        K and Cl.
    bs_layer_idx : 2-dimensional array of int
        containing zero-based indices of atoms of layers (from SF) defining
        the binding sites.
        e.g.
            bs_layer_idx[0] contains idx of atoms of layers forming the first
            boundary of the first binding site
            bs_layer_idx[1] contains idx of atoms of layers forming the second
            boundary of the first binding site
                            + the first boundary of the second binding site
    additional_bs_cutoff: float
        cutoff distance (in z-direction) in Angstrom for S0 and Scav.
    bs_radius: float
        threshold distance between center of BSs and particle in Angstrom for
        determining if particle is occupying.
        one of the BSs
    d_min_outer_cutoff: float
        threshold distance in Angstrom for determining if the particle is
        touching and thus occupying outer (the first or the last binding sites)
        BSs.

    Returns
    -------
    occupancy: list of lists
        # lists = # binding sites
        each of the lists contains (zero-based) indices of particles occupying
        the corresponding BS (S0 to Scav, 6 in total).
        occupancy[0] is particles occupancy S0
        occupancy[6] is particles occupying Scav
    """

    bs_layer_o_pos = np.array([positions[indices] for indices in sf_o_indices])
    bs_layer_o_pos_z = np.mean(bs_layer_o_pos, axis=1)[:, 2]
    sol_pos_z = positions[sol_indices][:, 2]

    # check if it is sorted in descending order which is needed
    # this ensures that first and second indices define the first BS (S0)
    assert all(np.sort(bs_layer_o_pos_z)[::-1] == bs_layer_o_pos_z)

    # bs_full_o_pos_z = np.array(list(bs_layer_o_pos_z) +
    #                          [bs_layer_o_pos_z[-1] - additional_bs_cutoff])

    bs_full_o_pos_z = np.array(
        list(bs_layer_o_pos_z) + [bs_layer_o_pos_z[-1] - additional_bs_cutoff]
    )

    # keep indices and pos_z only for sol within the range of bs_full_o_pos_z
    bound_idx = np.argwhere(
        (sol_pos_z < bs_full_o_pos_z[0]) & (sol_pos_z > bs_full_o_pos_z[-1])
    ).reshape(-1)
    sol_indices = sol_indices[bound_idx]
    sol_pos_z = sol_pos_z[bound_idx]

    # number of binding sites
    n_bs = len(bs_full_o_pos_z) - 1
    # sol_in_bs[i] == -1,  is unbound, is "above" the first BS (S0)
    # sol_in_bs[i] == j,  is in the j-th binding site
    # sol_in_bs[i] == n_bs ,  is below Scav
    sol_in_bs = np.searchsorted(-bs_full_o_pos_z, -sol_pos_z) - 1

    # compute channel axis
    # use oxygen from S0 and S1 (12 in total)
    axis_s0_end = np.mean(np.vstack(bs_layer_o_pos[0:3]), axis=0)
    # use oxygen from S3 and S4 (12 in total)
    axis_scav_end = np.mean(np.vstack(bs_layer_o_pos[3:6]), axis=0)
    axis = axis_s0_end - axis_scav_end
    axis /= np.linalg.norm(axis)

    occupancy = [[] for i in range(n_bs)]

    for sol_idx, sol_bs_i in zip(sol_indices, sol_in_bs):
        # skip if not inside SF
        if sol_bs_i < 0 or sol_bs_i > n_bs - 1:
            continue

        else:
            sol_vec = positions[sol_idx] - axis_scav_end
            sol_radial_vec = sol_vec - np.dot(axis, sol_vec) * axis
            r = np.linalg.norm(sol_radial_vec)

            if r < bs_radius:
                occupancy[sol_bs_i].append(sol_idx)

    return occupancy


def _check_flip(pos_all, sf_o_idx, cutoff=5):
    """check SF oxygen flips in a given frame

    Parameters
    ----------
    pos_all : narray
        positions of all atoms in the system
    sf_o_idx : narray
        indices for oxygen atoms in each layer
    cutoff : float
        cutoff distance for determining if a flip occurs

    Returns
    -------
    flips: array
        number of flips in each layer. flips[0] is the # flips for the first
        layer
    """
    flips = np.zeros(len(sf_o_idx), dtype=int)
    n_o = len(sf_o_idx[0])

    for i, layer in enumerate(sf_o_idx):
        pos = pos_all[layer]
        flip = [
            (np.linalg.norm(pos - pos[j], axis=1).sum() / (n_o - 1)) > cutoff
            for j in range(len(layer))
        ]
        flips[i] = np.sum(flip)

    return flips


def _compute_sf_diagonal(pos_all, sf_atom_idx):
    """calculate diagonal distance of SF atoms, e.g. O and CA

    Parameters
    ----------
    pos_all : narray
        positions of all atoms in the system
    sf_atom_idx : narray
        indices for atoms of interest in each layer. The first index and
        second index are in opposite pair. Same for the third and the fourth.

    Returns
    -------
    d: array
        all computed diagonal distances
    """
    diag_distances = []
    for _, indices in enumerate(sf_atom_idx):
        pos = pos_all[indices]
        diag_distances.append(np.linalg.norm(pos[0] - pos[1]))
        diag_distances.append(np.linalg.norm(pos[2] - pos[3]))
    return np.array(diag_distances)


def _compute_occupancy(k_occ_whole, w_occ_whole, bs_ignore=(0, 5)):
    """compute SF occupancy labels

    Parameters
    ----------
    k_occ_whole: list of lists of int
        k_occ_whole[i][j] contains the indicies of K+ bound in j-th BS for
        frame i.

    w_occ_whole: list of lists of int
        w_occ_whole[i][j] contains the indicies of water bound in j-th BS for
        frame i.

    bs_ignore: list of int
        contains indices of BSs from which no warning about double occupancy
        is reported.

    Returns
    -------
    occ_whole: narray
       occ_whole[i] is the occupancy label for frame i.

    occ_whole: list of tuples
       frame number and BS that double occupancy occurs.

    """

    n_bs = len(k_occ_whole[0])
    assert n_bs == 6

    occ_whole = []
    double_occ = []

    for t, (k_occ, w_occ) in enumerate(zip(k_occ_whole, w_occ_whole)):
        occ = ["0" for j in range(n_bs)]
        for i in range(len(occ)):
            if len(k_occ[i]) > 0 and len(w_occ[i]) > 0:
                occ[i] = "C"
                if i not in bs_ignore:
                    # print(f'At frame {t}, double occupancy in S{i}')
                    double_occ.append((t, i))
            elif len(k_occ[i]) > 0:
                occ[i] = "K"
            elif len(w_occ[i]) > 0:
                occ[i] = "W"
        occ_whole.append("".join(occ))

    occ_whole = np.array(occ_whole)
    return occ_whole, double_occ


def _compute_jumps(k_occ_all, w_occ_all):
    """compute ion and water net jump for the whole trajectory
        *** it ignores jump across S0 and Scav (S5) ***
        e.g. t_0 = [[],[],[],[],[],[]]
             t_1 = [[],[],[],[],[2],[]]
             jump = 1

    Parameters
    ----------
    k_occ_all: list of lists
        # lists = # binding sites
        each of the lists contains indices of ion occupying the corresponding
        binding site.

    w_occ_all: list of lists
        same as above, except that it is for water

    Returns
    -------
    jumps: narray
        jumps[i, 0] saves # net jumps of ion occurred between i-th step and
        (i+1)-th step.
        jumps[i, 1] saves # net jumps of water occurred between i-th step and
        (i+1)-th step.

    """

    jumps = np.zeros((len(k_occ_all) - 1, 2), dtype=int)
    for i in range(len(jumps)):
        jumps[i, 0] = _compute_jump(k_occ_all[i], k_occ_all[i + 1], t0=i)
        jumps[i, 1] = _compute_jump(w_occ_all[i], w_occ_all[i + 1], t0=i)
    return jumps


def _compute_jump(occ_t0, occ_t1, t0=0.0):
    """compute net number of jumps given the current and the next states.

        It ignores jump across S0 and Scav (S5) by setting the currently
        occupied BS as S0 and S5 if the ion/water is above S0 and below S5,
        respectively.

    Parameters
    ----------
    occ_t0 : list of lists of int
        each list contains atom idx that occupies the corresponding binding
        site at t=t0.
        e.g. occ_t0 = [[], [], [33577], [33596], [], []]
    occ_t1 : list of lists of int
        each list contains atom idx that occupies the corresponding binding
        site at t=t1.
        e.g. occ_t1 = [[33577], [], [], [33596], [], []]
    t0 : float (optional)
        current timestep, only used for error report/debug

    Returns
    -------
    jump: int
        net number of jumps given the current and the next occupation states
    """
    # only work for 6 binding sites
    assert len(occ_t0) == 6

    jump = 0
    checked = []

    # check occ_t1 for new positions of particles present in occ_t0
    for bs_i_t0, bs_t0 in enumerate(occ_t0):
        for sol_i_t0 in bs_t0:
            new_pos = [i for (i, bs_t1) in enumerate(
                occ_t1) if sol_i_t0 in bs_t1]
            if len(new_pos) > 1:
                raise AttributeError(f"At time {t0}, same solvent/solute \
is identified more than once")
            elif len(new_pos) == 0:
                if bs_i_t0 == 2 or bs_i_t0 == 3:
                    raise AttributeError(f"At step {t0}, new position of idx \
{sol_i_t0} in S{bs_i_t0} cannot be found, jump too much?")
                elif bs_i_t0 == 0 or bs_i_t0 == 1:
                    # assume it has escaped through S0, so set it beyond S0
                    new_pos_idx = 0
                else:
                    # assume it has escaped through Scav, so set it beyond Scav
                    new_pos_idx = 5
            else:
                new_pos_idx = new_pos[0]
            # jump > 0 means forward, toward S0
            jump += bs_i_t0 - new_pos_idx

            if sol_i_t0 not in checked:
                checked.append(sol_i_t0)
            else:
                raise AttributeError(
                    f"At step {t0}, {sol_i_t0} was found twice"
                    )

    for bs_i_t1, bs_t1 in enumerate(occ_t1):
        for sol_i_t1 in bs_t1:
            if sol_i_t1 in checked:
                continue
            else:
                if bs_i_t1 == 2 or bs_i_t1 == 3:
                    raise AttributeError(f"At step {t0+1}, history of idx \
{sol_i_t1} which is found in S2/S3 cannot be traced, jump too much?")
                elif bs_i_t1 == 0 or bs_i_t1 == 1:
                    # assume it has entered through S0, so set it at S0 to
                    # ignore jump between extracellular region and S0
                    old_pos_idx = 0
                else:
                    # assume it has entered through Scav, so set it at Scav to
                    # ignore jump between cytoplasm
                    # and Scav
                    old_pos_idx = 5
                jump += old_pos_idx - bs_i_t1
    return jump


@_count_time
def run(
    coor,
    traj,
    output="kperm",
    perm_count=("cross"),
    perm_details=False,
    sf_idx=None,
    sf_all_res=False,
    sf_diag=False,
    bs_radius=4.0,
):
    path = os.path.dirname(traj)

    log_loc = os.path.abspath(os.path.join(path, output + ".log"))

    logger = _create_logger(log_loc)

    print(f"Reading coordinate {coor}\nReading trajectory {traj}")
    u = mda.Universe(coor, traj, in_memory=False)

    if sf_idx is None:
        sf_idx = detect_sf(coor, quiet=True, all_res=sf_all_res)

    sf_o_idx = np.array([sf_idx["O"][i]["idx"]
                        for i in range(len(sf_idx["O"]))])
    sf_ca_idx = np.array([sf_idx["CA"][i]["idx"]
                         for i in range(len(sf_idx["CA"]))])

    k_idx, _, water_idx = _get_non_protein_index(coor)

    k_occupancy = []
    w_occupancy = []

    if perm_details:
        k_in_sf = []
        w_in_sf = []

    occupancy = np.zeros(len(u.trajectory), dtype="<6U")
    jumps = np.zeros((len(u.trajectory), 2), dtype=int)
    flips = np.zeros((len(u.trajectory), len(sf_o_idx)), dtype=int)

    if sf_diag:
        # start with first BS (S0) x2, end with the last BS x2
        o_d_diag = np.zeros(
            (len(u.trajectory), 2 * len(sf_o_idx)), dtype=float)
        ca_d_diag = np.zeros(
            (len(u.trajectory), 2 * len(sf_ca_idx)), dtype=float)

    for ts in u.trajectory:
        if sf_diag:
            o_d_diag[ts.frame] = _compute_sf_diagonal(ts.positions, sf_o_idx)
            ca_d_diag[ts.frame] = _compute_sf_diagonal(
                ts.positions, sf_ca_idx)

        flips[ts.frame] = _check_flip(ts.positions, sf_o_idx)

        k_occ = _find_bound(
            ts.positions, k_idx, sf_o_idx, bs_radius=bs_radius
        )
        w_occ = _find_bound(
            ts.positions, water_idx, sf_o_idx, bs_radius=bs_radius
        )

        k_occupancy.append(k_occ)
        w_occupancy.append(w_occ)

        if perm_details:
            # unwrap k_occ and w_occ for sf_occ_k.dat and sf_occ_w.dat
            k_occ_unwrap = [
                (index, bs_i) for bs_i, bs in enumerate(k_occ) for index in bs
            ]
            w_occ_unwrap = [
                (index, bs_i) for bs_i, bs in enumerate(w_occ) for index in bs
            ]

            for i, (index, bs_i) in enumerate(k_occ_unwrap):
                k_in_sf.append(
                    tuple(
                        [ts.frame, index, f"{ts.positions[index][2]:.5f}",
                         bs_i]
                        )
                )
            for i, (index, bs_i) in enumerate(w_occ_unwrap):
                w_in_sf.append(
                    tuple(
                        [ts.frame, index, f"{ts.positions[index][2]:.5f}",
                         bs_i]
                        )
                )

        if ts.frame % 1000 == 0:
            print(
                "\r" +
                f"Finished processing frame {ts.frame} / {len(u.trajectory)}",
                end=" ",
            )

    print("")
    occupancy[: len(k_occupancy)], double_occ = _compute_occupancy(
        k_occupancy, w_occupancy
    )
    if len(double_occ) > 0:
        logger.info(
            "Double occupancy in S1/S2/S3/S4 is found in %d frames. \
Check log file for details.", len(double_occ))
        for t, i in double_occ:
            logger.debug("In frame %d, double occupancy is found in S%d",
                         t, i)

    n_net_k_jump, n_net_w_jump, k_current_jump = "N/A", "N/A", "N/A"
    n_net_k_cross, n_net_w_cross, k_current_cross = "N/A", "N/A", "N/A"
    n_up_k_cross, n_down_k_cross = "N/A", "N/A"
    n_up_w_cross, n_down_w_cross = "N/A", "N/A"

    if "jump" in perm_count:
        jumps[: len(k_occupancy) -
              1] = _compute_jumps(k_occupancy, w_occupancy)
        k_j_sum = np.sum(jumps[:, 0])
        w_j_sum = np.sum(jumps[:, 1])

        n_net_k_jump = int(np.fix(k_j_sum / 5))
        n_net_w_jump = int(np.fix(w_j_sum / 5))

        k_current_jump = round(
            n_net_k_jump * 1.602e-19 / (u.trajectory.totaltime * 1e-12) * 1e12,
            6
        )  # unit: pA

    if "cross" in perm_count:
        kperm_traj, kperm_up, kperm_down = _count_perm_cross(
            k_occupancy, group1=[0, 1, 2], group2=[3, 4, 5]
        )
        wperm_traj, wperm_up, wperm_down = _count_perm_cross(
            w_occupancy, group1=[0, 1, 2], group2=[3, 4, 5]
        )

        if perm_details:
            _write_list_of_tuples(os.path.join(path, "kperm_up.dat"), kperm_up)
            _write_list_of_tuples(os.path.join(
                path, "kperm_down.dat"), kperm_down)
            _write_list_of_tuples(os.path.join(path, "wperm_up.dat"), wperm_up)
            _write_list_of_tuples(os.path.join(
                path, "wperm_down.dat"), wperm_down)

            _write_list_of_tuples(os.path.join(path, "k_in_sf.dat"), k_in_sf)
            _write_list_of_tuples(os.path.join(path, "w_in_sf.dat"), w_in_sf)

        n_net_k_cross = np.sum(kperm_traj)
        n_net_w_cross = np.sum(wperm_traj)
        n_up_k_cross = len(kperm_up)
        n_down_k_cross = len(kperm_down)
        n_up_w_cross = len(wperm_up)
        n_down_w_cross = len(wperm_down)

        k_current_cross = round(
            n_net_k_cross * 1.602e-19 / (u.trajectory.totaltime*1e-12) * 1e12,
            6
        )  # unit: pA

    logger.info("=================================")
    logger.info("Total time: %.3f ns", u.trajectory.totaltime/1e3)
    logger.info("dt: %.4f ns", u.trajectory.dt/1e3)
    logger.info("Number of K+: %d", len(k_idx))
    logger.info("Number of water: %d\n", len(water_idx))

    logger.info("Number of upward ion permeation events (cross) = %s",
                n_up_k_cross)
    logger.info("Number of downward ion permeation events (cross) = %s",
                n_down_k_cross)
    logger.info("Number of net ion permeation events (cross) = %s",
                n_net_k_cross)
    logger.info("Number of upward water permeation events (cross) = %s",
                n_up_w_cross)
    logger.info("Number of downward water permeation events (cross) = %s",
                n_down_w_cross)
    logger.info("Number of net water permeation events (cross) = %s",
                n_net_w_cross)
    logger.info("K+ current (cross) = %s pA\n", k_current_cross)

    logger.info("Number of net ion permeation events (jump) = %s",
                n_net_k_jump)
    logger.info("Number of net water permeation events (jump) = %s",
                n_net_w_jump)
    logger.info("K+ current (jump) = %s pA\n", k_current_jump)

    if sf_diag:
        data = np.hstack(
            (occupancy.reshape(-1, 1), jumps, flips, ca_d_diag, o_d_diag)
        ).astype("<8U")
        columns = (
            ["occupancy", "j_k", "j_w"] +
            [f"flip_{i}" for i in range(len(sf_o_idx))] +
            [f"{sf_idx['CA'][i]['name'][0]}_{sf_idx['CA'][i]['resid'][0]}_{j}"
             for i in range(len(sf_idx["CA"]))
             for j in range(2)] +
            [f"{sf_idx['O'][i]['name'][0]}_{sf_idx['O'][i]['resid'][0]}_{j}"
             for i in range(len(sf_idx["O"]))
             for j in range(2)]
        )
    else:
        data = np.hstack(
            (occupancy.reshape(-1, 1), jumps, flips)).astype("<8U")
        columns = ["occupancy", "j_k", "j_w"] + [
            f"flip_{i}" for i in range(len(sf_o_idx))
        ]

    data_loc = os.path.abspath(os.path.join(path, output + ".csv"))
    data = pd.DataFrame(data, columns=columns)
    data.to_csv(data_loc)
    logger.info("Results saved to %s", data_loc)
    logger.info("Log saved to %s", log_loc)
    logger.info("=================================")

    # remove all handlers to avoid multiple logging
    logger.handlers = []

    return data
