
#!/home/luke/miniconda3/envs/wlasl/bin/python
import argparse
from quewing2 import daemon, worker, retrieve_Data, store_Data

def run_daemon(args):
    daem = daemon()
    if args == 'watch':
        daem.start_n_watch()
    else:
        daem.start_n_monitor_simple()

def run_worker(args):
    wr = worker()
    info = retrieve_Data(wr.temp_path)
    if args == 'here':
        wr.start_here(info)

def main():
    parser = argparse.ArgumentParser(prog='quefeather.py')
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode', required=True)

    # Daemon subcommand
    daemon_parser = subparsers.add_parser('daemon', help='Run as daemon')

    daemon_parser.add_argument(
        'mode',
        choices=['watch', 'monitor'],
        help='Operation of daemon'
    )

    # Worker subcommand  
    worker_parser = subparsers.add_parser('worker', help='Run as worker')



    args = parser.parse_args()

              
    # run_train(args.verbose)




  