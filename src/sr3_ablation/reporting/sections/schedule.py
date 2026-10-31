# ADDED: new file, not part of the upstream SR3 codebase.
"""Sections of the Schedule Analysis: exponent, schedule tables and the schedule plot."""  # ADDED

from __future__ import annotations  # ADDED

import math  # ADDED
from collections.abc import Callable  # ADDED
from dataclasses import dataclass  # ADDED

from sr3_ablation.reporting.plots import plot_schedule_comparison  # ADDED
from sr3_ablation.reporting.sections.base import ReportContext  # ADDED
from sr3_ablation.reporting.tables import schedule_table, scientific, snr_at_timesteps_table  # ADDED

SCHEDULE_FIGURE_FILE = "schedule_comparison.png"  # ADDED

Number = float | int | None  # ADDED


def decimals(places: int) -> Callable[[Number], str]:  # ADDED
    """Formatter that prints a number with a fixed number of decimal places."""  # ADDED

    def format_number(value: Number) -> str:  # ADDED
        return f"{value:.{places}f}"  # ADDED

    return format_number  # ADDED


def timestep_or_never(value: Number) -> str:  # ADDED
    """Formatter for an optional timestep."""  # ADDED
    return "never" if value is None else str(int(value))  # ADDED


def scientific_number(value: Number) -> str:  # ADDED
    """Formatter for numbers spanning many orders of magnitude (wraps :func:`scientific`)."""  # ADDED
    return scientific(float(value))  # ADDED


def terminal_noise_level(context: ReportContext) -> float:  # ADDED
    """sqrt(alpha_bar'_T): the signal coefficient of x_T and the lowest noise level Run B sees."""  # ADDED
    return math.sqrt(context.modified.summary().alpha_bar_last)  # ADDED


def snr_power_law_at_t(context: ReportContext) -> float:  # ADDED
    """Approximation SNR'(T) ~ (x + 1) / (T * beta_T ** x) for a near-linear base schedule."""  # ADDED
    schedule = context.config.schedule  # ADDED
    return (context.exponent + 1) / (schedule.n_timestep * schedule.linear_end**context.exponent)  # ADDED


@dataclass(frozen=True)  # ADDED
class ScheduleNumberSection:  # ADDED
    """One number derived from the schedules, quoted inline in the Schedule Analysis prose."""  # ADDED

    placeholder: str  # ADDED
    value: Callable[[ReportContext], Number]  # ADDED
    formatter: Callable[[Number], str]  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        return self.formatter(self.value(context))  # ADDED


SCHEDULE_NUMBER_SECTIONS: tuple[ScheduleNumberSection, ...] = (  # ADDED
    ScheduleNumberSection(  # ADDED
        "ORIGINAL_ALPHA_BAR_T",  # ADDED
        lambda c: c.original.summary().alpha_bar_last,  # ADDED
        scientific_number,  # ADDED
    ),  # ADDED
    ScheduleNumberSection(  # ADDED
        "ORIGINAL_NOISE_LEVEL_MIN",  # ADDED
        lambda c: math.sqrt(c.original.summary().alpha_bar_last),  # ADDED
        decimals(3),  # ADDED
    ),  # ADDED
    ScheduleNumberSection(  # ADDED
        "ORIGINAL_SNR_CROSSING",  # ADDED
        lambda c: c.original.first_t_snr_below_one(),  # ADDED
        timestep_or_never,  # ADDED
    ),  # ADDED
    ScheduleNumberSection(  # ADDED
        "MODIFIED_BETA_FIRST",  # ADDED
        lambda c: c.modified.summary().beta_first,  # ADDED
        scientific_number,  # ADDED
    ),  # ADDED
    ScheduleNumberSection(  # ADDED
        "MODIFIED_BETA_T",  # ADDED
        lambda c: c.modified.summary().beta_last,  # ADDED
        scientific_number,  # ADDED
    ),  # ADDED
    ScheduleNumberSection(  # ADDED
        "MODIFIED_ALPHA_BAR_T",  # ADDED
        lambda c: c.modified.summary().alpha_bar_last,  # ADDED
        decimals(4),  # ADDED
    ),  # ADDED
    ScheduleNumberSection("MODIFIED_SNR_T", lambda c: c.modified.summary().snr_last, decimals(1)),  # ADDED
    ScheduleNumberSection("SNR_POWER_LAW_T", snr_power_law_at_t, decimals(0)),  # ADDED
    ScheduleNumberSection("XT_SIGNAL_COEF", terminal_noise_level, decimals(3)),  # ADDED
    ScheduleNumberSection(  # ADDED
        "XT_NOISE_COEF",  # ADDED
        lambda c: math.sqrt(1.0 - c.modified.summary().alpha_bar_last),  # ADDED
        decimals(3),  # ADDED
    ),  # ADDED
    ScheduleNumberSection("EMBEDDING_SHIFT", lambda c: 1.0 - terminal_noise_level(c), decimals(3)),  # ADDED
)  # ADDED


class ExponentSection:  # ADDED
    """The declared exponent x, to two decimals."""  # ADDED

    placeholder = "EXPONENT"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        return f"{context.exponent:.2f}"  # ADDED


class ScheduleTableSection:  # ADDED
    """theta_1, theta_T, alpha_bar_T and SNR(T) of both schedules."""  # ADDED

    placeholder = "SCHEDULE_TABLE"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        return schedule_table(context.original, context.modified)  # ADDED


class TrajectorySnrTableSection:  # ADDED
    """SNR of both schedules at the saved trajectory timesteps."""  # ADDED

    placeholder = "TRAJECTORY_SNR_TABLE"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        return snr_at_timesteps_table(  # ADDED
            context.original,  # ADDED
            context.modified,  # ADDED
            context.config.sampling.trajectory_timesteps,  # ADDED
        )  # ADDED


class ScheduleFigureSection:  # ADDED
    """theta_t and theta_t ** x against t on the same axes."""  # ADDED

    placeholder = "SCHEDULE_FIGURE"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        path = plot_schedule_comparison(  # ADDED
            context.config.schedule,  # ADDED
            context.exponent,  # ADDED
            context.config.sampling.trajectory_timesteps,  # ADDED
            context.assets / SCHEDULE_FIGURE_FILE,  # ADDED
        )  # ADDED
        return context.image("Original and power-transformed schedules", path)  # ADDED
