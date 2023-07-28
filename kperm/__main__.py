#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import argparse
import argcomplete
from pathlib import Path
import kperm as kp


def sf(channel):
    channel.detect_sf()


def run(channel):
    channel.run(perm_details=True, perm_count=['cross', 'jump'])


def stats(channel):
    channel.compute_stats()


def main():
    # eval "$(register-python-argcomplete kperm)"

    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("-s", help="coordinate file", required=True)
    parser.add_argument("-f", nargs='+',
                        help="trajectories or  folders containing log files",
                        required=True)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.mode == 'sf':

        print()
        channel = kp.Channel()
        channel.set_coord(args.s)
        channel.set_trajs(args.f)
        print()

        sf(channel)
    elif args.mode == 'run':

        print()
        channel = kp.Channel()
        channel.set_coord(args.s)
        channel.set_trajs(args.f)
        print()

        run(channel)

    elif args.mode == 'stats':

        print()
        channel = kp.Channel()
        channel.set_coord(args.s)

        if isinstance(args.f, str):
            args.f = [args.f]
        
        log_paths = []
        traj_paths = []

        for path in args.f:
            if Path(path).is_dir():
                log_paths.append(path)
            else:
                traj_paths.append(path)
                log_paths.append(Path(path).parent)

        if len(traj_paths) > 0:
            channel.set_trajs(traj_paths)

        channel.load(log_paths)
        print()

        stats(channel)



if __name__ == "__main__":
    main()
