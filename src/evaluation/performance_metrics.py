from dataclasses import dataclass
from typing import Optional


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

    #thread_utilization: float  db_insertion_time_total / (total_time × thread_count)
    #thread_efficiency_score: float something with cpu
    #lock_contention: float db_lock_wait_time / db_insertion_time_total

    system_cores_avg: float

    db_insertion_time_total: float
    db_lock_wait_time: float
