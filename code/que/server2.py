from multiprocessing.managers import BaseManager
from .core import que
import time
import signal
import sys
import os

class QueueManager(BaseManager): 
    pass

# Global reference to allow graceful shutdown
que_instance = None
server_instance = None

def connect_que(max_retries=5, retry_delay=2) -> que:
    QueueManager.register('get_que')
    
    for _ in range(max_retries):
        try:
            m = QueueManager(address=('localhost', 50000), authkey=b'abracadabra')
            m.connect()
            return m.get_que()  # type: ignore
        except ConnectionRefusedError:
            print(f"Queue server not ready, retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            
    raise RuntimeError(
        "Cannot connect to queue server. "
        "Start it with: python que_server.py"
    )

def graceful_shutdown(save_state=True):
    """Save state and shutdown server"""
    global que_instance, server_instance
    
    print("\n🛑 Shutting down server...")
    
    if que_instance and save_state:
        try:
            print("💾 Saving queue state...")
            que_instance.save_state()
            print("✅ State saved successfully")
        except Exception as e:
            print(f"❌ Error saving state: {e}")
    
    if server_instance:
        try:
            server_instance.stop_event.set()
        except:
            pass
    
    sys.exit(0)

def reload_handler(signum, frame):
    """Handle reload signal (SIGUSR1)"""
    global que_instance
    
    print("\n🔄 Reload signal received (SIGUSR1)...")
    
    # Save current state
    if que_instance:
        try:
            que_instance.save_state()
            print("✅ State saved for reload")
        except Exception as e:
            print(f"❌ Error saving state: {e}")
            return
    
    # Restart the process
    print("♻️  Restarting server process...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

def shutdown_handler(signum, frame):
    """Handle interrupt signals (SIGINT, SIGTERM)"""
    graceful_shutdown(save_state=True)

def main():
    global que_instance, server_instance
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, shutdown_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, shutdown_handler)  # kill command
    signal.signal(signal.SIGUSR1, reload_handler)    # Reload trigger
    
    print("🚀 Starting queue server...")
    
    # Initialize queue (will load saved state automatically)
    que_instance = que()
    print("✅ Queue initialized (state loaded if available)")
    
    # Register and start server
    QueueManager.register('get_que', callable=lambda: que_instance)
    m = QueueManager(address=('localhost', 50000), authkey=b'abracadabra')
    server_instance = m.get_server()
    
    print("✨ Server running on localhost:50000")
    print(f"   PID: {os.getpid()}")
    print("   Press Ctrl+C to shutdown gracefully")
    print(f"   Send SIGUSR1 to reload: kill -USR1 {os.getpid()}\n")
    
    try:
        server_instance.serve_forever()
    except KeyboardInterrupt:
        graceful_shutdown(save_state=True)

if __name__ == '__main__':
    main()