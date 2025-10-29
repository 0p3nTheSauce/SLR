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
        
    def run_daemon(self, setting: Literal['sWatch', 'sMonitor','monitorO', 'idle', 'idle_log'], args : Optional[argparse.Namespace] = None):
        daem = daemon()
        print("The daemon sais: ")
        cmd_d = {
            0: 'sWatch',
            1: 'sMonitor',
            2: 'monitorO',
            3: 'idle',
            4: 'idle_log'
        }
        
        
        if setting == cmd_d[0]:
            daem.start_n_watch()
        elif setting == cmd_d[1]:
            daem.start_n_monitor()
        elif setting == cmd_d[2]:
            daem.monitor_log()
        elif setting == cmd_d[3]:
            daem.start_idle()
        elif setting == cmd_d[4]:
            daem.start_idle_log()
        else:
            print('huh?')
            print(f'You gave me: {setting}')
            print(f'but i only accept: {cmd_d.values()}')

    def run_worker(self, setting: Literal['work', 'idle', 'idle_log', 'idle_gpu'], args : Optional[argparse.Namespace] = None):
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
        elif setting == 'idle_gpu':
            z = wr.sim_gpu_usage()
            print(z.shape)
        else:
            print('huh?')
            
def main():
    parser = argparse.ArgumentParser(prog='quefeather.py')
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode', required=True)

    # Daemon subcommand
    daemon_parser = subparsers.add_parser('daemon', help='Run as daemon')

    daemon_parser.add_argument(
        'setting',
        choices=['sWatch', 'sMonitor','monitorO', 'idle', 'idle_log'],
        help='Operation of daemon'
    )

    # Worker subcommand  
    worker_parser = subparsers.add_parser('worker', help='Run as worker')

    worker_parser.add_argument(
        'setting',
        choices=['work', 'idle', 'idle_log', 'idle_gpu'],
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




  