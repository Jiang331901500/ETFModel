import os

class StopInput():
    FLAG_FILE = "stop_loop.flag"
    def should_stop(self):
        return os.path.exists(self.FLAG_FILE)

    def clean_resources(self):
        if os.path.exists(self.FLAG_FILE):
            os.remove(self.FLAG_FILE)
    
    def __del__(self):
        self.clean_resources()

# Usage example:
if __name__ == "__main__":
    import time
    stop_input = StopInput()
    while True:
        time.sleep(1)
        if stop_input.should_stop():
            print("Stopping input loop as flag file exists.")
            break
        else:
            print("Continuing input loop as flag file does not exist.")
    