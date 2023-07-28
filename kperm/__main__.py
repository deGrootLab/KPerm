#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import argparse
import argcomplete
from pathlib import Path
import kperm as kp


def sf(channel):
    channel.detect_sf()


def run(channel, perm_count, check_water, check_flip):
    channel.run(perm_details=True, perm_count=perm_count, 
                check_water=check_water,check_flip=check_flip)


def stats(channel):
    channel.compute_stats()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=['sf', 'run', 'stats'],
                        help='detect SF atoms, count permeation events,' +
                        'and compute summary of permeation events ' +
                        'in selected trajectories')
    parser.add_argument("-s", help="coordinate file", required=True)
    parser.add_argument("-f", nargs='+',
                        help="trajectories or  folders containing log files",
                        required=True)
    parser.add_argument("--jump", help='count permeation events ' +
                        'by number of jumps', action="store_true")
    parser.add_argument("--noW", help='ignore water', action="store_true")
    parser.add_argument("--noFlip", help='do not check oxygem flip', 
                        action="store_true")

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

        if args.jump:
            run(channel, ('cross', 'jump'), check_water=(not args.noW),
                check_flip=(not args.noFlip))
        else:
            run(channel, ('cross'), check_water=(not args.noW),
                check_flip=(not args.noFlip))

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
