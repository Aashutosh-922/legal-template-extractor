from threading import Lock


class MetricsStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._total_runs = 0
        self._failed_runs = 0
        self._successful_runs = 0
        self._total_retries = 0
        self._total_hallucination_flags = 0
        self._total_missing_fields = 0
        self._required_coverage_sum = 0.0

    def record_success(
        self,
        retries_used: int,
        hallucination_flags: int,
        missing_fields: int,
        required_coverage: float,
    ) -> None:
        with self._lock:
            self._total_runs += 1
            self._successful_runs += 1
            self._total_retries += retries_used
            self._total_hallucination_flags += hallucination_flags
            self._total_missing_fields += missing_fields
            self._required_coverage_sum += required_coverage

    def record_failure(self) -> None:
        with self._lock:
            self._total_runs += 1
            self._failed_runs += 1

    def summary(self) -> dict:
        with self._lock:
            parse_success_rate = (
                float(self._successful_runs) / float(self._total_runs) if self._total_runs else 0.0
            )
            average_retries = (
                float(self._total_retries) / float(self._successful_runs) if self._successful_runs else 0.0
            )
            hallucination_rate = (
                float(self._total_hallucination_flags) / float(self._successful_runs)
                if self._successful_runs
                else 0.0
            )
            average_required_coverage = (
                float(self._required_coverage_sum) / float(self._successful_runs)
                if self._successful_runs
                else 0.0
            )

            return {
                "total_runs": self._total_runs,
                "successful_runs": self._successful_runs,
                "failed_runs": self._failed_runs,
                "parse_success_rate": parse_success_rate,
                "average_retries_per_success": average_retries,
                "hallucination_rate": hallucination_rate,
                "average_required_coverage": average_required_coverage,
                "total_missing_fields": self._total_missing_fields,
            }
