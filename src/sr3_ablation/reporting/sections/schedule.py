# ADDED: new file, not part of the upstream SR3 codebase.
"""Sections of the Schedule Analysis: exponent, schedule tables and the schedule plot."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from sr3_ablation.reporting.plots import plot_schedule_comparison
from sr3_ablation.reporting.sections.base import ReportContext
from sr3_ablation.reporting.tables import schedule_table, scientific, snr_at_timesteps_table

SCHEDULE_FIGURE_FILE = "schedule_comparison.png"

Number = float | int | None


def decimals(places: int) -> Callable[[Number], str]:
    """Formatter that prints a number with a fixed number of decimal places."""

    def format_number(value: Number) -> str:
        return f"{value:.{places}f}"

    return format_number


def timestep_or_never(value: Number) -> str:
    """Formatter for an optional timestep."""
    return "never" if value is None else str(int(value))


def scientific_number(value: Number) -> str:
    """Formatter for numbers spanning many orders of magnitude (wraps :func:`scientific`)."""
    return scientific(float(value))


def terminal_noise_level(context: ReportContext) -> float:
    """sqrt(alpha_bar'_T): the signal coefficient of x_T and the lowest noise level Run B sees."""
    return math.sqrt(context.modified.summary().alpha_bar_last)


def snr_power_law_at_t(context: ReportContext) -> float:
    """Approximation SNR'(T) ~ (x + 1) / (T * beta_T ** x) for a near-linear base schedule."""
    schedule = context.config.schedule
    return (context.exponent + 1) / (schedule.n_timestep * schedule.linear_end**context.exponent)


@dataclass(frozen=True)
class ScheduleNumberSection:
    """One number derived from the schedules, quoted inline in the Schedule Analysis prose."""

    placeholder: str
    value: Callable[[ReportContext], Number]
    formatter: Callable[[Number], str]

    def render(self, context: ReportContext) -> str:
        return self.formatter(self.value(context))


SCHEDULE_NUMBER_SECTIONS: tuple[ScheduleNumberSection, ...] = (
    ScheduleNumberSection(
        "ORIGINAL_ALPHA_BAR_T",
        lambda c: c.original.summary().alpha_bar_last,
        scientific_number,
    ),
    ScheduleNumberSection(
        "ORIGINAL_NOISE_LEVEL_MIN",
        lambda c: math.sqrt(c.original.summary().alpha_bar_last),
        decimals(3),
    ),
    ScheduleNumberSection(
        "ORIGINAL_SNR_CROSSING", lambda c: c.original.first_t_snr_below_one(), timestep_or_never
    ),
    ScheduleNumberSection(
        "MODIFIED_BETA_FIRST", lambda c: c.modified.summary().beta_first, scientific_number
    ),
    ScheduleNumberSection(
        "MODIFIED_BETA_T", lambda c: c.modified.summary().beta_last, scientific_number
    ),
    ScheduleNumberSection(
        "MODIFIED_ALPHA_BAR_T", lambda c: c.modified.summary().alpha_bar_last, decimals(4)
    ),
    ScheduleNumberSection("MODIFIED_SNR_T", lambda c: c.modified.summary().snr_last, decimals(1)),
    ScheduleNumberSection("SNR_POWER_LAW_T", snr_power_law_at_t, decimals(0)),
    ScheduleNumberSection("XT_SIGNAL_COEF", terminal_noise_level, decimals(3)),
    ScheduleNumberSection(
        "XT_NOISE_COEF",
        lambda c: math.sqrt(1.0 - c.modified.summary().alpha_bar_last),
        decimals(3),
    ),
    ScheduleNumberSection("EMBEDDING_SHIFT", lambda c: 1.0 - terminal_noise_level(c), decimals(3)),
)


class ExponentSection:
    """The declared exponent x, to two decimals."""

    placeholder = "EXPONENT"

    def render(self, context: ReportContext) -> str:
        return f"{context.exponent:.2f}"


class ScheduleTableSection:
    """theta_1, theta_T, alpha_bar_T and SNR(T) of both schedules."""

    placeholder = "SCHEDULE_TABLE"

    def render(self, context: ReportContext) -> str:
        return schedule_table(context.original, context.modified)


class TrajectorySnrTableSection:
    """SNR of both schedules at the saved trajectory timesteps."""

    placeholder = "TRAJECTORY_SNR_TABLE"

    def render(self, context: ReportContext) -> str:
        return snr_at_timesteps_table(
            context.original, context.modified, context.config.sampling.trajectory_timesteps
        )


class ScheduleFigureSection:
    """theta_t and theta_t ** x against t on the same axes."""

    placeholder = "SCHEDULE_FIGURE"

    def render(self, context: ReportContext) -> str:
        path = plot_schedule_comparison(
            context.config.schedule,
            context.exponent,
            context.config.sampling.trajectory_timesteps,
            context.assets / SCHEDULE_FIGURE_FILE,
        )
        return context.image("Original and power-transformed schedules", path)
