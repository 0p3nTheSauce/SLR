
#!/home/luke/miniconda3/envs/wlasl/bin/python
import argparse
from quewing import daemon, worker, WR_NAME, DN_NAME
from typing import Literal, TypeAlias
Feather: TypeAlias = Literal["worker", "daemon"]
FEATHERS = [WR_NAME, DN_NAME]

class queFeather:
    def __init__(
        self,
        mode: Feather,
    ):
        self.mode = mode
    
    def run(self,args):
        if self.mode == "daemon":
            self.run_daemon(args)
        else:
            self.run_worker(args)
        
    def run_daemon(self, setting: Literal['watch', 'monitor', 'idle']):
        daem = daemon()
        if setting == 'watch':
            daem.start_n_watch()
        elif setting == 'monitor':
            daem.start_n_monitor()
        else:
            daem.start_idle()

    def run_worker(self, setting: Literal['work', 'idle']):
        wr = worker()
        if setting == 'work':
            wr.work()
        else:
            wr.idle("Testing")

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
        choices=['work', 'idle'],
        help='Operation of worker'
    )

    args = parser.parse_args()

    qf = queFeather(mode=args.mode)
    qf.run(args)
    
if __name__ == '__main__':
    main()




  