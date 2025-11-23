import curses
import time
import threading
import os
from collections import deque

# --- CONFIGURATION ---
LOG_FILE_PATH = "app.log"

# --- SHARED STATE ---
state = {
    # A deque is like a list, but if we add a 51st item, 
    # the 1st item is automatically removed.
    "logs": deque(maxlen=50), 
    "running": True,
    "file_status": "Waiting for file..."
}

def log_tail_worker():
    """
    Background thread that mimics the unix 'tail -f' command.
    """
    # 1. Wait for file to exist
    while state["running"] and not os.path.exists(LOG_FILE_PATH):
        state["file_status"] = "File not found (waiting)..."
        time.sleep(1)

    state["file_status"] = "Tailing file..."

    # 2. Open and Tail
    try:
        with open(LOG_FILE_PATH, "r") as f:
            # Move pointer to the end of the file (seek_end)
            # Remove this line if you WANT to read the whole file from start
            f.seek(0, 2) 
            
            while state["running"]:
                line = f.readline()
                if line:
                    # If we found a line, clean it and add to state
                    state["logs"].append(line.strip())
                else:
                    # If no new line, wait briefly (don't burn CPU)
                    time.sleep(0.1)
    except Exception as e:
        state["file_status"] = f"Error: {str(e)}"

def draw_main_menu(stdscr):
    stdscr.addstr(0, 0, "=== SERVER DASHBOARD ===", curses.A_BOLD)
    stdscr.addstr(2, 0, "System is running normally.")
    
    # Show a snippet of the latest log just for flavor
    last_log = state["logs"][-1] if state["logs"] else "No logs yet..."
    stdscr.addstr(4, 0, f"Latest Event: {last_log}", curses.A_DIM)

    stdscr.addstr(6, 0, "Controls:")
    stdscr.addstr(7, 2, "[Ctrl + n] : Open Live Log Monitor")
    stdscr.addstr(8, 2, "[q]        : Quit")

def draw_log_monitor(stdscr):
    height, width = stdscr.getmaxyx()
    
    # Header
    header = f"=== LIVE LOGS ({LOG_FILE_PATH}) ==="
    stdscr.addstr(0, 0, header, curses.A_BOLD)
    stdscr.addstr(0, width - 20, "[Ctrl+x] Back", curses.A_REVERSE)
    
    status_msg = f"Status: {state['file_status']}"
    stdscr.addstr(1, 0, status_msg, curses.A_DIM)
    
    # Draw a separator line
    stdscr.hline(2, 0, curses.ACS_HLINE, width)
    
    # --- RENDER LOGS ---
    # We can't print more lines than the screen height allows.
    # We reserve 3 lines for header, so max lines = height - 3
    max_display_lines = height - 3
    
    # Get the most recent logs that fit on screen
    # list(state['logs']) converts the deque to a list we can slice
    current_logs = list(state["logs"])
    visible_logs = current_logs[-max_display_lines:]
    
    for idx, log_line in enumerate(visible_logs):
        # Ensure line fits width to prevent crashing on wrap
        clean_line = log_line[:width-1] 
        
        # Color code based on content (Simple Syntax Highlighting)
        color = curses.A_NORMAL
        if "ERROR" in clean_line:
            color = curses.A_BOLD | curses.A_STANDOUT # Highlight Errors
        elif "WARNING" in clean_line:
            color = curses.A_BOLD
            
        stdscr.addstr(3 + idx, 0, clean_line, color)

def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    
    # Start the background thread
    worker_thread = threading.Thread(target=log_tail_worker, daemon=True)
    worker_thread.start()
    
    current_view = "main"

    try:
        while True:
            stdscr.erase()
            
            if current_view == "main":
                draw_main_menu(stdscr)
            elif current_view == "monitor":
                draw_log_monitor(stdscr)

            stdscr.refresh()

            # Input Handling
            try:
                key = stdscr.getch()
            except:
                key = -1

            if key == ord('q'):
                break
            elif key == 14: # Ctrl + n
                current_view = "monitor"
            elif key == 24: # Ctrl + x
                current_view = "main"
            
            time.sleep(0.05)
            
    finally:
        state["running"] = False
        worker_thread.join()

curses.wrapper(main)