#!/home/luke/miniconda3/envs/wlasl/bin/python

import argparse
from quewing import daemon, worker, WR_NAME, DN_NAME
from typing import Literal, TypeAlias

Feather: TypeAlias = Literal["worker", "daemon"]
FEATHERS = [WR_NAME, DN_NAME]


class QueFeather:
    """Dispatcher for daemon and worker operations."""
    
    def __init__(self, mode: Feather):
        self.mode = mode
        self._daemon = None
        self._worker = None
    
    @property
    def daemon_instance(self) -> daemon:
        """Lazy initialization of daemon."""
        if self._daemon is None:
            self._daemon = daemon()
        return self._daemon
    
    @property
    def worker_instance(self) -> worker:
        """Lazy initialization of worker."""
        if self._worker is None:
            self._worker = worker()
        return self._worker
    
    def run(self, args: argparse.Namespace):
        """Main entry point - dispatches to appropriate handler."""
        if self.mode == "daemon":
            self._run_daemon(args)
        else:
            self._run_worker(args)
    
    def _run_daemon(self, args: argparse.Namespace):
        """Handle daemon operations."""
        print("The daemon says:")
        
        daem = self.daemon_instance
        setting = args.setting
        
        # Handle recovery mode
        if args.recover:
            self._handle_recovery(daem, setting, args.run_id)
            return
        
        # Dispatch to appropriate daemon method
        daemon_operations = {
            'sWatch': daem.start_n_watch,
            'sMonitor': daem.start_n_monitor,
            'monitorO': daem.monitor_log,
            'idle': daem.start_idle,
            'idle_log': daem.start_idle_log,
        }
        
        operation = daemon_operations.get(setting)
        if operation:
            operation()
        else:
            self._invalid_setting(setting, daemon_operations.keys())
    
    def _handle_recovery(self, daem: daemon, setting: str, run_id: str):
        """Handle recovery mode for daemon."""
        available_settings = ['sWatch', 'sMonitor']
        if setting not in available_settings:
            raise ValueError(
                f"Setting '{setting}' is not available for recovery. "
                f"Available: {', '.join(available_settings)}"
            )
        daem.recover(setting, run_id)
    
    def _run_worker(self, args: argparse.Namespace):
        """Handle worker operations."""
        print("The worker says:")
        
        wr = self.worker_instance
        setting = args.setting
        
        # Dispatch to appropriate worker method
        if setting == 'work':
            wr.work()
        elif setting == 'idle':
            wr.idle("Testing", args.wait, args.cycles)
        elif setting == 'idle_log':
            wr.idle_log("Testing", args.wait, args.cycles)
        elif setting == 'idle_gpu':
            z = wr.sim_gpu_usage()
            print(z.shape)
        else:
            worker_operations = ['work', 'idle', 'idle_log', 'idle_gpu']
            self._invalid_setting(setting, worker_operations)
    
    @staticmethod
    def _invalid_setting(setting: str, valid_options):
        """Handle invalid setting errors."""
        print('Invalid setting!')
        print(f'You gave: {setting}')
        print(f'Valid options: {", ".join(valid_options)}')


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog='quefeather.py',
        description='Queue management system for daemon and worker processes'
    )
    subparsers = parser.add_subparsers(
        dest='mode',
        help='Operation mode',
        required=True
    )

    # Daemon subcommand
    daemon_parser = subparsers.add_parser(
        'daemon',
        help='Run as daemon'
    )
    daemon_parser.add_argument(
        'setting',
        choices=['sWatch', 'sMonitor', 'monitorO', 'idle', 'idle_log'],
        help='Daemon operation to perform'
    )
    daemon_parser.add_argument(
        '-re', '--recover',
        action='store_true',
        help='Recover from run failure'
    )
    daemon_parser.add_argument(
        '-ri', '--run_id',
        type=str,
        default=None,
        help='Run ID for recovery (uses temp file if not specified)'
    )

    # Worker subcommand
    worker_parser = subparsers.add_parser(
        'worker',
        help='Run as worker'
    )
    worker_parser.add_argument(
        'setting',
        choices=['work', 'idle', 'idle_log', 'idle_gpu'],
        help='Worker operation to perform'
    )
    worker_parser.add_argument(
        '-w', '--wait',
        type=int,
        default=None,
        help='Idle time in seconds'
    )
    worker_parser.add_argument(
        '-c', '--cycles',
        type=int,
        default=None,
        help='Number of idle cycles'
    )

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    qf = QueFeather(mode=args.mode)
    qf.run(args)


if __name__ == '__main__':
    main()