"""Runtime policy for concurrency, timeouts and adversarial review."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchConfig:
    max_concurrency: int = 4
    task_timeout_seconds: float = 45.0
    global_timeout_seconds: float = 240.0
    batch_failure_threshold: float = 0.5
    max_replans: int = 1
    max_review_rounds: int = 4
    review_pass_score: float = 0.78
    min_citation_coverage: float = 0.80
    max_pass_issue_severity: int = 1
    max_verification_queries_per_round: int = 4
    max_blue_patches_per_round: int = 20
    max_patch_growth_chars: int = 4_000
    max_blue_patch_generation_attempts: int = 2
    enable_semantic_claim_verification: bool = True
    min_claim_support_rate: float = 0.80
    min_score_improvement: float = 0.015
    oscillation_window: int = 4
    enable_dynamic_swarm: bool = True
    dynamic_claim_support_drop_tolerance: float = 0.02
    max_swarm_agents: int = 5
    max_worker_invocations: int = 16
    max_stagnant_review_rounds: int = 2

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.task_timeout_seconds <= 0 or self.global_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if not 0 <= self.batch_failure_threshold <= 1:
            raise ValueError("batch_failure_threshold must be between 0 and 1")
        if not 0 <= self.review_pass_score <= 1:
            raise ValueError("review_pass_score must be between 0 and 1")
        if not 0 <= self.min_citation_coverage <= 1:
            raise ValueError("min_citation_coverage must be between 0 and 1")
        if self.max_review_rounds < 1:
            raise ValueError("max_review_rounds must be >= 1")
        if not 0 <= self.max_pass_issue_severity <= 3:
            raise ValueError("max_pass_issue_severity must be between 0 and 3")
        if self.max_verification_queries_per_round < 0:
            raise ValueError("max_verification_queries_per_round must be >= 0")
        if self.max_blue_patches_per_round < 1:
            raise ValueError("max_blue_patches_per_round must be >= 1")
        if self.max_patch_growth_chars < 0:
            raise ValueError("max_patch_growth_chars must be >= 0")
        if self.max_blue_patch_generation_attempts < 1:
            raise ValueError("max_blue_patch_generation_attempts must be >= 1")
        if not 0 <= self.min_claim_support_rate <= 1:
            raise ValueError("min_claim_support_rate must be between 0 and 1")
        if not 0 <= self.dynamic_claim_support_drop_tolerance <= 1:
            raise ValueError("dynamic_claim_support_drop_tolerance must be between 0 and 1")
        if self.max_swarm_agents < 1:
            raise ValueError("max_swarm_agents must be >= 1")
        if self.max_worker_invocations < 1:
            raise ValueError("max_worker_invocations must be >= 1")
        if self.max_stagnant_review_rounds < 1:
            raise ValueError("max_stagnant_review_rounds must be >= 1")
