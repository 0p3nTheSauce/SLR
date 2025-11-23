from que.core import que
from que.server import connect_que
import json

def test_dump_peak():
    #works fine
    q = que()
    run = q._peak_run('cur_run', 0)
    print(json.dumps(run, indent=4))
    
def test_dump_peak_server():
    #AttributeError: 'AutoProxy[get_que]' object has no attribute '_peak_run'
    q = connect_que()
    run = q._peak_run('cur_run', 0)
    print(json.dumps(run, indent=4))
    
if __name__ == '__main__':
    # test_dump_peak()
    test_dump_peak_server()