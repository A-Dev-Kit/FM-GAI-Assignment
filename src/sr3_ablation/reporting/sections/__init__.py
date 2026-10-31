# ADDED: new file, not part of the upstream SR3 codebase.
"""Report sections, one class per template placeholder.  # ADDED

To add a section, implement :class:`ReportSection` and append an instance to  # ADDED
``SECTION_REGISTRY``; neither the builder nor the other sections change.  # ADDED
"""  # ADDED

from sr3_ablation.reporting.sections.analysis import (  # ADDED
    ContributionTableSection,  # ADDED
    QualitativeAnalysisSection,  # ADDED
    SupplementaryAnalysisSection,  # ADDED
    TakeawaySection,  # ADDED
)  # ADDED
from sr3_ablation.reporting.sections.base import ReportContext, ReportSection, pending  # ADDED
from sr3_ablation.reporting.sections.dataset import DatasetTableSection  # ADDED
from sr3_ablation.reporting.sections.diagnostics import StartDiagnosticSection  # ADDED
from sr3_ablation.reporting.sections.implementation import CodeDiffSection, ScheduleEvidenceSection  # ADDED
from sr3_ablation.reporting.sections.results import (  # ADDED
    ComparisonFigureSection,  # ADDED
    ResultsTableSection,  # ADDED
    Run0SamplesSection,  # ADDED
)  # ADDED
from sr3_ablation.reporting.sections.schedule import (  # ADDED
    SCHEDULE_NUMBER_SECTIONS,  # ADDED
    ExponentSection,  # ADDED
    ScheduleFigureSection,  # ADDED
    ScheduleTableSection,  # ADDED
    TrajectorySnrTableSection,  # ADDED
)  # ADDED
from sr3_ablation.reporting.sections.setup import EnvironmentSection, SetupTableSection  # ADDED
from sr3_ablation.reporting.sections.training import (  # ADDED
    CheckpointComparisonSection,  # ADDED
    LossFigureSection,  # ADDED
    LossTableSection,  # ADDED
    SnapshotFiguresSection,  # ADDED
    ValidationFigureSection,  # ADDED
)  # ADDED
from sr3_ablation.reporting.sections.trajectories import TrajectoryFigureSection  # ADDED

SECTION_REGISTRY: tuple[ReportSection, ...] = (  # ADDED
    ExponentSection(),  # ADDED
    *SCHEDULE_NUMBER_SECTIONS,  # ADDED
    ScheduleTableSection(),  # ADDED
    TrajectorySnrTableSection(),  # ADDED
    ScheduleFigureSection(),  # ADDED
    DatasetTableSection(),  # ADDED
    SetupTableSection(),  # ADDED
    EnvironmentSection(),  # ADDED
    CodeDiffSection(),  # ADDED
    ScheduleEvidenceSection(),  # ADDED
    Run0SamplesSection(),  # ADDED
    LossFigureSection(),  # ADDED
    LossTableSection(),  # ADDED
    ValidationFigureSection(),  # ADDED
    SnapshotFiguresSection(),  # ADDED
    CheckpointComparisonSection(),  # ADDED
    TrajectoryFigureSection(),  # ADDED
    ResultsTableSection(),  # ADDED
    ComparisonFigureSection(),  # ADDED
    StartDiagnosticSection(),  # ADDED
    QualitativeAnalysisSection(),  # ADDED
    SupplementaryAnalysisSection(),  # ADDED
    TakeawaySection(),  # ADDED
    ContributionTableSection(),  # ADDED
)  # ADDED

__all__ = ["SECTION_REGISTRY", "ReportContext", "ReportSection", "pending"]  # ADDED
