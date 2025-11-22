from multiprocessing.managers import BaseManager

class MathsClass:
    def add(self, x, y):
        return x + y
    def mul(self, x, y):
        return x * y

class MyManager(BaseManager):
    pass

MyManager.register('Maths', MathsClass)

if __name__ == '__main__':
    with MyManager() as manager:
        maths = manager.Maths() #type:  ignore
        print(maths.add(4, 3))         # prints 7
        print(maths.mul(7, 8))         # prints 56
        
# from multiprocessing.managers import BaseManager
# from queue import Queue
# queue = Queue()
# class QueueManager(BaseManager): pass
# QueueManager.register('get_queue', callable=lambda:queue)
# m = QueueManager(address=('', 50000), authkey=b'abracadabra')
# s = m.get_server()
# s.serve_forever()