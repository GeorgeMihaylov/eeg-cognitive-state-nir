"""Application AutoML for per-user portfolio and adaptation orchestration.

Scientific fold-local optimization belongs to ``bench.automl.scientific``. This package
must receive an already separated application calibration/validation view and
must never select a candidate from outer-test results.

The core package (this directory) has no dependency on any specific
dataset, sensor modality, or ML framework. To use it on a concrete
project:

1. Implement `ModelAdapter` for each model family you want to offer
   (sklearn estimators already satisfy it; wrap torch models so they
   also expose `freeze_all_but_head` / `unfreeze_all` if you want
   HEAD_ONLY support).
2. Build a `CandidateRegistry` listing (model, adaptation method,
   governance status) tuples for your project.
3. Write a `tags` vocabulary for `SubjectMetaFeatures` (e.g. device,
   channel layout, recording site — whatever distinguishes your users'
   data) and use it in `CandidateSpec.required_tags` / `.condition`.
4. Instantiate `PersonalizedAutoML` with your registry, metric
   function, and (optionally) shadow diagnostic runners for methods
   still under confirmatory review.

The optional :mod:`bench.automl.personalized.bindings` module connects these
framework-independent contracts to the canonical project model APIs.
"""

from .adaptation import adapt_candidate, score_candidate
from .meta_features import SubjectMetaFeatures, extract_meta_features
from .orchestrator import PersonalizedAutoML
from .portfolio import PortfolioRecord, PortfolioStore
from .protocols import ConditionFn, MetricFn, ModelAdapter, ModelBuilder, ShadowDiagnosticRunner
from .registry import AdaptationMethod, CandidateRegistry, CandidateSpec, CandidateStatus
from .search import SearchResult, staged_search
from .splits import InnerSplit, build_inner_split

__all__ = [
    "AdaptationMethod",
    "CandidateRegistry",
    "CandidateSpec",
    "CandidateStatus",
    "ConditionFn",
    "InnerSplit",
    "MetricFn",
    "ModelAdapter",
    "ModelBuilder",
    "PersonalizedAutoML",
    "PortfolioRecord",
    "PortfolioStore",
    "SearchResult",
    "ShadowDiagnosticRunner",
    "SubjectMetaFeatures",
    "adapt_candidate",
    "build_inner_split",
    "extract_meta_features",
    "score_candidate",
    "staged_search",
]
