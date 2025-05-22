import threading
import time
import sys

def spinner(message: str="Thinking"):
    stop = False

    def spin():
        symbols = ['|', '/', '-', '\\']
        idx = 0
        while not stop:
            sys.stdout.write(f"\r{message} {symbols[idx % len(symbols)]}")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)

    thread = threading.Thread(target=spin)
    thread.start()

    def stop_spinner():
        nonlocal stop
        stop = True
        thread.join()
        sys.stdout.write('\r' + ' ' * 40 + '\r')  # Clear line

    return stop_spinner
