# ADDED: new file, not part of the upstream SR3 codebase.
"""Report sections, one class per template placeholder.

To add a section, implement :class:`ReportSection` and append an instance to
``SECTION_REGISTRY``; neither the builder nor the other sections change.
"""

from sr3_ablation.reporting.sections.base import ReportContext, ReportSection, pending
from sr3_ablation.reporting.sections.dataset import DatasetTableSection
from sr3_ablation.reporting.sections.diagnostics import StartDiagnosticSection
from sr3_ablation.reporting.sections.implementation import CodeDiffSection, ScheduleEvidenceSection
from sr3_ablation.reporting.sections.results import (
    ComparisonFigureSection,
    ResultsTableSection,
    Run0SamplesSection,
)
from sr3_ablation.reporting.sections.schedule import (
    SCHEDULE_NUMBER_SECTIONS,
    ExponentSection,
    ScheduleFigureSection,
    ScheduleTableSection,
    TrajectorySnrTableSection,
)
from sr3_ablation.reporting.sections.setup import EnvironmentSection, SetupTableSection
from sr3_ablation.reporting.sections.training import (
    CheckpointComparisonSection,
    LossFigureSection,
    LossTableSection,
    SnapshotFiguresSection,
    ValidationFigureSection,
)
from sr3_ablation.reporting.sections.trajectories import TrajectoryFigureSection

SECTION_REGISTRY: tuple[ReportSection, ...] = (
    ExponentSection(),
    *SCHEDULE_NUMBER_SECTIONS,
    ScheduleTableSection(),
    TrajectorySnrTableSection(),
    ScheduleFigureSection(),
    DatasetTableSection(),
    SetupTableSection(),
    EnvironmentSection(),
    CodeDiffSection(),
    ScheduleEvidenceSection(),
    Run0SamplesSection(),
    LossFigureSection(),
    LossTableSection(),
    ValidationFigureSection(),
    SnapshotFiguresSection(),
    CheckpointComparisonSection(),
    TrajectoryFigureSection(),
    ResultsTableSection(),
    ComparisonFigureSection(),
    StartDiagnosticSection(),
)

__all__ = ["SECTION_REGISTRY", "ReportContext", "ReportSection", "pending"]
