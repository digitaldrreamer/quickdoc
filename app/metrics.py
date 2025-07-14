import time
import psutil
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ResourceTracker:
    """A context manager to track resource usage for a block of code."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.start_cpu_time: Optional[float] = None
        self.end_cpu_time: Optional[float] = None
        self.start_mem_rss: Optional[int] = None
        self.end_mem_rss: Optional[int] = None
        self.processing_method: Optional[str] = None

    def __enter__(self):
        """Start tracking resources."""
        self.start_time = time.monotonic()
        try:
            self.start_cpu_time = self.process.cpu_times().user
            self.start_mem_rss = self.process.memory_info().rss
        except psutil.NoSuchProcess:
            logger.warning("Process not found, skipping resource tracking.")
            self.start_cpu_time = 0.0
            self.start_mem_rss = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking resources."""
        self.end_time = time.monotonic()
        try:
            self.end_cpu_time = self.process.cpu_times().user
            self.end_mem_rss = self.process.memory_info().rss
        except psutil.NoSuchProcess:
            logger.warning("Process not found, skipping resource tracking.")
            self.end_cpu_time = self.start_cpu_time
            self.end_mem_rss = self.start_mem_rss

    def log_method(self, method: str):
        """Log the primary processing method used."""
        self.processing_method = method

    def get_metrics(self, file_size_bytes: int) -> Dict[str, Any]:
        """Return a dictionary of all tracked metrics."""
        if self.start_time is None or self.end_time is None or \
           self.start_cpu_time is None or self.end_cpu_time is None or \
           self.start_mem_rss is None or self.end_mem_rss is None:
            return {"error": "Tracking was not properly started or stopped."}

        duration_ms = (self.end_time - self.start_time) * 1000
        cpu_time_ms = (self.end_cpu_time - self.start_cpu_time) * 1000
        mem_diff_mb = (self.end_mem_rss - self.start_mem_rss) / (1024 * 1024)

        return {
            "processing_duration_ms": round(duration_ms, 2),
            "cpu_time_ms": round(cpu_time_ms, 2),
            "memory_usage_mb": round(self.end_mem_rss / (1024 * 1024), 2),
            "memory_diff_mb": round(mem_diff_mb, 2),
            "file_size_bytes": file_size_bytes,
            "processing_method": self.processing_method or "unknown",
        } 