#!/home/luke/miniconda3/envs/wlasl/bin/python

import argparse
from quewing import daemon, worker, WR_NAME, DN_NAME
from typing import Literal, TypeAlias, Optional
Feather: TypeAlias = Literal["worker", "daemon"]
FEATHERS = [WR_NAME, DN_NAME]

class queFeather:
    def __init__(
        self,
        mode: Feather,
    ):
        self.mode = mode
    
    def run(self,args, kwargs : Optional[argparse.Namespace] = None):
        if self.mode == "daemon":
            self.run_daemon(args, kwargs)
        else:
            self.run_worker(args, kwargs)
        
    def run_daemon(self, setting: Literal['watch', 'monitor', 'idle', 'idle_mon'], args : Optional[argparse.Namespace] = None):
        daem = daemon()
        print("The daemon sais: ")
        if setting == 'watch':
            daem.start_n_watch()
        elif setting == 'monitor':
            daem.start_n_monitor()
        elif setting == 'idle':
            daem.start_idle()
        else:
            print('huh?')

    def run_worker(self, setting: Literal['work', 'idle', 'idle_log'], args : Optional[argparse.Namespace] = None):
        wr = worker()
        print("The worker sais: ")
        wait = args.wait if args else None
        cycles = args.cycles if args else None
        
        if setting == 'work':
            wr.work()
        elif setting == 'idle':
            wr.idle("Testing", wait, cycles)
        elif setting == 'idle_log':
            wr.idle_log("Testing", wait, cycles)
        else:
            print('huh?')
def main():
    parser = argparse.ArgumentParser(prog='quefeather.py')
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode', required=True)

    # Daemon subcommand
    daemon_parser = subparsers.add_parser('daemon', help='Run as daemon')

    daemon_parser.add_argument(
        'setting',
        choices=['watch', 'monitor', 'idle'],
        help='Operation of daemon'
    )

    # Worker subcommand  
    worker_parser = subparsers.add_parser('worker', help='Run as worker')

    worker_parser.add_argument(
        'setting',
        choices=['work', 'idle', 'idle_log'],
        help='Operation of worker'
    )
    
    worker_parser.add_argument(
        '-w',
        '--wait',
        help='Idle time (seconds)',
        type=int,
        default= None
    )
    
    worker_parser.add_argument(
        '-c',
        '--cycles',
        help = 'How many loops',
        type = int,
        default = None
    )

    args = parser.parse_args()

    qf = queFeather(mode=args.mode)
    qf.run(args.setting, args)
    
if __name__ == '__main__':
    main()




  