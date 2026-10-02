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

**que --host <server_ip> [options] [command ...]**

#### Available options:
-   `--host`          Host IP or hostname to connect to
-   `--ssh_user`      SSH username (default: current user)
-   `--ssh_key`       Path to SSH private key. Tries to use ~/.ssh/id_rsa or ed25519 by default
-   `--port_client`   Local port for SSH tunnel (default: 50000)
-   `--port_server`   Remote port on server (default: 50000)
-   `--max_retries`   Max connection retries (default: 5)
-   `--retry_delay`   Seconds between retries (default: 2)
-   `--yes`, `-y`     Answer yes to confirmation prompts (see below)

If running the que-shell on the server, run:

```bash
que --host 'localhost'
```

otherwise if connecting remotely:

```bash
que --host '123.456.78.910' #example IP address
```

after the first use the last host will be used by default.

#### Running a single command

Anything after the options is run as one QueShell command, and `que` then exits instead of
opening the shell. The exit status is 0 if the command succeeded, and 1 if it failed or was
cancelled, so this works in scripts:

```bash
que server status
que list to_run
que daemon set_sweep --sweep_path configfiles/sweeps/S3D/exp007/config.yaml
```

- Options for `que` itself go before the command. Everything after the command is its own
  arguments, as in the shell.
- Connection messages go to stderr, so stdout is just the command's output. (The
  "Connecting to last-used host" lines come from the `que` wrapper that `setup.sh` generates,
  so re-run `setup.sh` once to move them to stderr too.)
- Commands that ask for confirmation (`clear`, `remove`, `logs -c`, and `create`/`add` of a
  duplicate run) refuse when there's no terminal to ask on. Pass `--yes` to answer yes:
  `que --yes clear fail_runs`.
- One-shot mode doesn't show the banner or read/write `~/.que_shell_history`.

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

When using `create` a new training run is specified using the same parser as *training.py* (see [training](../../README.md#training)) and is added to `to_run`. `add` also uses this parser, and is used to add already completed runs to `old_runs`

`to_run` and `cur_run` only hold runs without results or errors, so a completed or failed run can't be `move`d back into them: requeue it with `recover`, or `copy` it with `--clean_slate`.

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

When there are no runs in `to_run` and no sweep trials left to hand out, the Daemon idles, checking for new work periodically. It stops when the `daemon stop` command is used. Flags can be used to send a stop signal to the supervisor, or the worker.

#### Recovery

If there is an outside influence (power failure) the que-training service will autorecover, if it was awake before. 

Nothing has to be saved by hand for this: every change to the Que is written to `Runs.json` before it returns, and the server state (`Server.json`: sweep, daemon settings) is written whenever it changes and after each worker exits. Both are written atomically, so a crash mid-write leaves the previous file intact. A sweep's progress is not stored but counted from the Que (its trials in `old_runs`), so it can't drift from it; on startup, a difference from the saved count is logged as a warning.

Otherwise, If a run fails, the `recover` command can be used.  In the event of an exception, specify the location as `fail_runs`:

```bash
(que)$ recover -ol fail
```

#### State history

Every version of `Runs.json` and `Server.json` is kept in a git repo of their own, in `state/`, so the project's history isn't flooded with Que changes. A server installed by `setup.sh` runs `que-training-state-backup.timer`, which commits both files every 15 minutes if they changed (the message counts the runs in each location), then pushes if the repo has a remote. The snapshots from `save -t` and `old_ques/` stay untracked. See [state_backup.py](./state_backup.py).

To back up off the machine, create an empty **private** repo (the Que holds every run's config), then, from the repo root:

```bash
python -m src.que.state_backup init --remote git@github.com:<user>/<repo>.git
```

Pushes run unattended, so they need an SSH key without a passphrase (or a running agent). A failed push keeps the commit, and the next run pushes it. `systemctl status que-training-state-backup` shows the last result, and `journalctl -u que-training-state-backup` the history.

To restore an earlier version, find it in the log, copy it out, and `load` it (which makes it the saved state):

```bash
git -C src/que/state log --format='%h %ad %s' --date=iso -- Runs.json
git -C src/que/state show <commit>:Runs.json > /tmp/Runs_restore.json
que load que -ip /tmp/Runs_restore.json
```

#### Files

The Que's data and logs are kept apart from the code, in gitignored directories:
- `state/`: `Runs.json` (the Que), `Server.json` (server state), timestamped snapshots from `save -t`, and `old_ques/` (archived Ques). `state/` is also a git repo of its own, versioning `Runs.json` and `Server.json` (see [State history](#state-history))
- `logs/`: `Server.log` (server, daemon and worker) and `Training.log` (training and testing output). A server installed by `setup.sh` rotates them with logrotate (weekly, or sooner past 50 MB; 8 kept, older ones gzipped)

Until 2026-10-02 these lived directly in `src/que/`. The server moves them into place when it starts (`migrate_legacy_files` in `core.py`), never overwriting a file already there. On a machine that doesn't run the server (e.g. to read `Runs.json` from `src/results`), run it once by hand, from `src/`, and only while no Que server on that machine is still running the old code:

```bash
python -c "from src.que.core import migrate_legacy_files; print(*migrate_legacy_files(), sep='\n')"
```

#### Misc

- `attach` attaches to tmux session (only opens on the shell side)
- `wandb` open up wandb website
- `logs` Follow the server's logs, read through the server so it works over the SSH tunnel too: `-s` for `Server.log`, `-t` for `Training.log` (`-c` clears instead). `-j` follows this machine's systemd journal (requires sudo).
- `save` Save a copy of the que or server state to a .json file (state is already saved automatically, see [Recovery](#recovery)). `-t` timestamps the file name; `save all -t` snapshots both as a matching `Runs_<ts>.json`/`Server_<ts>.json` pair.
- `load` Load the que or server state from a .json file (`-ip`, default: the server's own), which then becomes the saved state
