import statistics
import threading
import time
import psutil


class ResourceMonitor:

    def __init__(self):
        self.monitor_thread = None
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []

    def start_monitoring(self):
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []

        def monitor():
            while self.monitoring:
                self.cpu_samples.append(psutil.cpu_percent())
                self.memory_samples.append(psutil.virtual_memory().used / 1024 / 1024)  # MB
                time.sleep(0.5)

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)

        return {
            'cpu_avg': statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            'memory_peak': max(self.memory_samples) if self.memory_samples else 0
        }
    