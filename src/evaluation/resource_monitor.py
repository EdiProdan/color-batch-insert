import statistics
import threading
import psutil

class ResourceMonitor:

    def __init__(self):
        self.monitor_thread = None
        self.monitoring = False
        self.cpu_samples = []


    def start_monitoring(self):
        self.monitoring = True
        self.cpu_samples = []

        # warm-up psutil
        psutil.cpu_percent(interval=None, percpu=True)

        def monitor():
            while self.monitoring:
                # system CPU as “core equivalents”
                per_cpu = psutil.cpu_percent(interval=0.5, percpu=True)
                self.cpu_samples.append(sum(per_cpu) / 100.0)

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)

        def avg(xs): return statistics.mean(xs) if xs else 0

        return {
            'system_cores_avg': avg(self.cpu_samples),
        }
