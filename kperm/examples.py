from pathlib import Path
import urllib.request


def download_charge_scaling():
    data_root_url = (
        "https://github.com/deGrootLab/KPerm/"
        "raw/main/examples/charge-scaling/"
    )
    pdb = Path("MthK.pdb")
    if pdb.is_file():
        print(str(pdb) + " exists, will not download it.")
    else:
        urllib.request.urlretrieve(data_root_url + "MthK.pdb", pdb)

    for i in range(3):
        name = f"traj-{i}"
        Path(name).mkdir(exist_ok=True)

        traj = Path(name, name + ".xtc")
        if traj.is_file():
            print(str(traj) + " exists, will not download it.")
        else:
            urllib.request.urlretrieve(
                data_root_url + f"{name}/{name}.xtc", traj)

    print("Finished downloading MthK.pdb and trajectories.")
