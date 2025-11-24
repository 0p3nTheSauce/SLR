from que.core import que, QueEmpty
from que.server import connect_manager
import json
import subprocess
import sys
import logging

logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	)
logger = logging.getLogger(__name__)

def test_dump_peak():
    #works fine
    q = que(logger)
    run = q.peak_run('cur_run', 0)
    print(json.dumps(run, indent=4))
    
def test_dump_peak_server():
    #works fine with non underscore method
    man = connect_manager()
    q = man.get_que()
    run = q.peak_run('to_run', 0)
    print(json.dumps(run, indent=4))
    
def test_subprocess():
    proc = subprocess.Popen(
        ['python', 'extable.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = proc.communicate()
    print("STDOUT:")
    print(stdout)
    print("STDERR:")
    print(stderr)
    
def test_subrocess2():
    proc = subprocess.Popen(
        [sys.executable, '-u', '-m', 'que.daemon'],
        stdout=open('./que/Daemon.log', 'a'),
        stderr=subprocess.STDOUT,
        bufsize=0
    )
    return_code = proc.wait()
    
def test_ex():
    try:
        raise QueEmpty('cur_run')
    except Exception as e:
        print(e)

def test_clear():
    man = connect_manager()
    q = man.get_que()
    try:
        q.clear_runs('cur_run')
    except Exception as e:
        print(e)
        
def test_clear2():
    q = que(logger)
    try:
        q.clear_runs('cur_run')
    except Exception as e:
        print(e)

if __name__ == '__main__':
    # test_dump_peak()
    # test_dump_peak_server()
    # test_subprocess()
    # test_subrocess2()
    # test_ex()
    test_clear()
    pass