from dataclasses import dataclass


@dataclass
class PerformanceMetrics:

    algorithm_name: str
    scenario: str
    run_number: int

    total_time: float
    throughput: float
    success_rate: float

    processing_overhead: float
    conflicts: int
    retries: int

    memory_peak: float
    cpu_avg: float
