from dataclasses import dataclass


@dataclass
class PerformanceMetrics:

    algorithm_name: str
    scenario: str
    run_number: int
    thread_count: int
    batch_size: int

    total_time: float
    throughput: float
    success_rate: float

    processing_overhead: float
    conflicts: int

    system_cores_avg: float

    db_insertion_time_total: float
    db_lock_wait_time: float
