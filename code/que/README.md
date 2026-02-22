# Que 

The Que feature (pronounced queue) allows users to create training runs, and add them to a queue, as well as some other helper functions like changing position etc. 

## Table of Contents

- [Setup](#setup)
- [Removal](#removal)

## Setup

The recommended approach is to run [setup.sh](./setup.sh).
```bash
cd que
chmod +x ./setup.sh
sudo ./setup.sh
```

The script will prompt the user whether to set up the `client` side, or `server` side:

* `Server`:

    Selecting the server option creates a systemd service: `que-training` and binds the command `que` to open the shell. This hosts the `Que Server` locally. The server needs to use the `wlasl` conda environment for the training runs. The user will be prompted to specify server initialisation flags. 

* `Client`:

    Only the `que` shell command is created. The user will be prompted to specify the conda environement, either `wlasl` or `wlasl_cpu`. The shell can be used to remotely connect to the server with ssh tunneling.

Under the hood `que` is configured to point to [shell.py](./shell.py) and the `que-training` service runs [server.py](./server.py).

## Removal

The actions of [setup.sh](./setup.sh) can be undone with [unsetup.sh](./unsetup.sh)
```bash
chmod +x ./unsetup.sh
sudo ./unsetup.sh
```

## Usage

### The `que` command

**que --host <server_ip> [options]**

#### Available options:
-   `--host`          Host IP or hostname to connect to
-   `--ssh_user`      SSH username (default: current user)
-   `--ssh_key`       Path to SSH private key. Tries to use ~/.ssh/id_rsa or ed25519 by default
-   `--port_client`   Local port for SSH tunnel (default: 50000)
-   `--port_server`   Remote port on server (default: 50000)
-   `--max_retries`   Max connection retries (default: 5)
-   `--retry_delay`   Seconds between retries (default: 2)

If running the que-shell on the server, run:

```bash
que --host 'localhost'
```

otherwise if connecting remotely:

```bash
que --host '123.456.78.910' #example IP address
```

after the first use the last host will be used by default.

### The QueShell

```bash
╔═══════════════════════════════════════╗
║          QueShell                     ║
║   Queue Management System             ║
╚═══════════════════════════════════════╝

Type help or ? to list commands.

(que)$ help

                     Available Commands                     
╭──────────────┬───────────────────────────────────────────╮
│ Command      │ Description                               │
├──────────────┼───────────────────────────────────────────┤
│ create       │ Create a new training run                 │
│ add          │ Add a completed training run to old_runs  │
│ remove       │ Remove a run                              │
│ clear        │ Clear runs                                │
│ list         │ List runs                                 │
│ quit         │ Exit queShell                             │
│ shuffle      │ Reposition a run                          │
│ move         │ Move run between locations                │
│ edit         │ Edit run                                  │
│ display      │ Display run config                        │
│ daemon       │ Interact with the worker process          │
│ server       │ Interact with the server context          │
│ worker       │ Interact with the worker                  │
│ logs         │ Interact with Que log files               │
│ load         │ Load the state of the Que or Daemon       │
│ save         │ Save the state of the Que or Daemon       │
│ recover      │ Recover a failed run                      │
│ wandb        │ Open the wandb page for a run, or project │
╰──────────────┴───────────────────────────────────────────╯

Tip: Use 'help <command>' for detailed information about a specific command
```

The QueShell has a number of different features:

#### Que Management:

Internally, the Que is composed of the following locations:
- to_run/new/tr
- cur_run/cur/cr
- old_runs/old/or
- fail_runs/fail/fr

Each of these locations can be viewed with `list` command. Alternatively a single run can be viewed in a particular location with the `display` command. 

When using `create` a new training run is specified using the same parser as *training.py* (see [training](../../README.md#training)) and is added to `to_run`. 

#### Que Daemon

To start the training que, use the command: `daemon start`


When the Daemon is started, is spawns a supervisor process and a worker process. 

- ##### Supervisor: 
    If awake, starts the worker, waits for it, then repeat.
- ##### Worker: 
    takes training runs from `to_run` and moves them to `cur_run` and performs training and testing. 

The status of which can be checked with the command `server status`:
```bash
                     Server Status                     
  Server                  Running (PID: 2954960)       
  Daemon                   Awake:           ✓          
                           Stop on Fail:    ✓          
                           Supervisor PID:  2958175    
  Worker                   Task:            training   
                           Run ID:          sz27ivxv   
                           Worker PID:      2958274 
```

Once training and testing are complete, the completed run with results is added to `old_runs`. If an exception occurs, the failed run is added to `fail_runs`. If `stop on fail` is *True*, then the Que Daemon will halt training. Otherwise it will continue. This can be set when being prompted during [setup](#setup). 

The Daemon will stop when it reaches the end of the queue, or when when the `daemon stop` command is used. Flags can be used to send a stop signal to the supervisor, or the worker. 

#### Recovery

If there is an outside influence (power failure) the que-training service will autorecover, if it was awake before. 

Otherwise, If a run fails, the `recover` command can be used.  In the event of an exception, specify the location as `fail_runs`:

```bash
(que)$ recover -ol fail
```

#### Misc

- `attach` attaches to tmux session (only opens on the shell side)
- `wandb` open up wandb website
- `logs` View the logs from the worker, or the server (not inclusive of systemd service logs).
- `save` Save state of que or server to .json file
- `load` Load state of que or server from .json file
