# SPDX-FileCopyrightText: PyPSA-Earth and PyPSA-Eur Authors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compare existing 2023 hydropower validation networks."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pypsa

from build_hydro_station_mapping import build_component_mapping
from validate_hydro_generation import validate_component_generation


RUNS = {
    "No cascade 0.175": "config.zm.valid_2023_nocascade_hydro_corrected",
    "Cascade 0.175": "config.zm.valid_2023_cascade_hydro_corrected",
    "Cascade 0.210": "config.zm.valid_2023_cascade_hydro_m021",
    "Cascade 0.250": "config.zm.valid_2023_cascade_hydro_m025",
}

REFERENCE_DIR = Path("data/hydro_reference")

KAFUE_GROUP = "Kafue Gorge Upper and Lower"

KAFUE_MAPPING = {
    "Kafue Gorge Upper": KAFUE_GROUP,
    "Kafue Gorge Lower": KAFUE_GROUP,
}


def run_validation(label, run, zambia_root, output_dir, metadata):
    """Validate an existing solved network without reoptimizing it."""
    directory = output_dir / run
    directory.mkdir(parents=True, exist_ok=True)

    network_path = (
        zambia_root / "results" / run
        / "networks" / "elec_s_10_ec_lv1.0_3h.nc"
    )

    n = pypsa.Network(str(network_path))

    assert n.snapshots.year.unique().tolist() == [2023]

    mapping = build_component_mapping(n, metadata)
    mapping.to_csv(directory / "station_mapping.csv", index=False)

    inputs = SimpleNamespace(
        ref_annual=REFERENCE_DIR / "ERB_annual_generation_2023.csv",
        ref_monthly=REFERENCE_DIR / "ERB_monthly_generation_2023.csv",
    )

    outputs = SimpleNamespace(
        annual_csv=directory / "annual_comparison.csv",
        annual_png=directory / "annual_comparison.png",
        monthly_csv=directory / "monthly_comparison.csv",
        monthly_png=directory / "monthly_comparison.png",
    )

    validate_component_generation(
        n,
        mapping,
        SimpleNamespace(input=inputs, output=outputs),
    )

    annual = pd.read_csv(outputs.annual_csv, index_col=0)
    annual.index.name = "station_name"
    annual = annual.reset_index().assign(scenario=label)

    monthly = pd.read_csv(outputs.monthly_csv, parse_dates=["month"])
    monthly = monthly.assign(scenario=label)

    return annual, monthly


def aggregate_groups(df, group_kafue=False):
    """Aggregate references and model generation consistently."""
    df = df.copy()

    if group_kafue:
        df["station_name"] = df["station_name"].replace(KAFUE_MAPPING)

    keys = ["scenario", "station_name"]

    if "month" in df.columns:
        keys.append("month")

    result = (
        df.groupby(keys, sort=False)[["model_gwh", "reference_gwh"]]
        .sum(min_count=1)
        .reset_index()
    )

    result["diff_gwh"] = result["model_gwh"] - result["reference_gwh"]
    result["diff_pct"] = (
        100 * result["diff_gwh"]
        / result["reference_gwh"].where(result["reference_gwh"] != 0)
    )

    return result


def calculate_metrics(df):
    """Calculate errors only for observations with available references."""
    records = []

    for scenario, group in df.groupby("scenario", sort=False):
        observed = group.dropna(subset=["reference_gwh", "model_gwh"])

        reference = observed["reference_gwh"].sum()
        error = observed["diff_gwh"]

        records.append({
            "scenario": scenario,
            "observations": len(observed),
            "model_gwh": observed["model_gwh"].sum(),
            "reference_gwh": reference,
            "bias_pct": 100 * error.sum() / reference,
            "wape_pct": 100 * error.abs().sum() / reference,
            "mae_gwh": error.abs().mean(),
        })

    return pd.DataFrame(records)


def plot_annual(df, labels, path):
    model = df.pivot(
        index="station_name",
        columns="scenario",
        values="model_gwh",
    )

    reference = (
        df.loc[df["scenario"].eq(labels[0])]
        .set_index("station_name")["reference_gwh"]
    )

    model = model.reindex(columns=labels)
    model.insert(0, "ERB 2023", reference)

    ax = model.plot.bar(figsize=(12, 6), width=0.8)

    ax.set_xlabel("")
    ax.set_ylabel("Electricity generation (GWh)")
    ax.set_title("Annual hydropower generation — 2023")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_monthly(df, labels, path):
    stations = df["station_name"].unique()

    fig, axes = plt.subplots(
        len(stations), 1,
        figsize=(11, 3 * len(stations)),
        sharex=True,
        squeeze=False,
    )

    for station, ax in zip(stations, axes[:, 0]):
        subset = df.loc[df["station_name"].eq(station)]

        reference = (
            subset.loc[subset["scenario"].eq(labels[0])]
            .dropna(subset=["reference_gwh"])
            .sort_values("month")
        )

        ax.plot(
            reference["month"],
            reference["reference_gwh"],
            color="black",
            marker="o",
            linewidth=2,
            label="ERB 2023",
        )

        for label in labels:
            model = subset.loc[
                subset["scenario"].eq(label)
            ].sort_values("month")

            ax.plot(
                model["month"],
                model["model_gwh"],
                marker=".",
                label=label,
            )

        ax.set_title(station)
        ax.set_ylabel("GWh")
        ax.grid(alpha=0.25)

    axes[0, 0].legend(ncol=2)

    fig.suptitle("Monthly hydropower generation — 2023")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zambia-root",
        type=Path,
        default=Path("../pypsa-zambia"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hydro_validation/sensitivity_2023"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(
        REFERENCE_DIR / "ERB_station_metadata_2023.csv"
    )

    annual_parts = []
    monthly_parts = []

    for label, run in RUNS.items():
        annual, monthly = run_validation(
            label, run, args.zambia_root, args.output_dir, metadata
        )

        annual_parts.append(annual)
        monthly_parts.append(monthly)

    annual = pd.concat(annual_parts, ignore_index=True)
    monthly = pd.concat(monthly_parts, ignore_index=True)

    comparisons = {
        "cascading_effect": (
            ["No cascade 0.175", "Cascade 0.175"],
            True,
        ),
        "inflow_sensitivity": (
            ["Cascade 0.175", "Cascade 0.210", "Cascade 0.250"],
            False,
        ),
    }

    for name, (labels, group_kafue) in comparisons.items():
        annual_comparison = aggregate_groups(
            annual.loc[annual["scenario"].isin(labels)],
            group_kafue,
        )

        monthly_comparison = aggregate_groups(
            monthly.loc[monthly["scenario"].isin(labels)],
            group_kafue,
        )

        annual_metrics = calculate_metrics(annual_comparison)
        monthly_metrics = calculate_metrics(monthly_comparison)

        for suffix, table in {
            "annual": annual_comparison,
            "monthly": monthly_comparison,
            "annual_metrics": annual_metrics,
            "monthly_metrics": monthly_metrics,
        }.items():
            table.to_csv(
                args.output_dir / f"{name}_{suffix}.csv",
                index=False,
            )

        plot_annual(
            annual_comparison,
            labels,
            args.output_dir / f"{name}_annual.png",
        )

        plot_monthly(
            monthly_comparison,
            labels,
            args.output_dir / f"{name}_monthly.png",
        )

        print(f"\n=== {name.upper()}: ANNUAL ===")
        print(annual_metrics.round(3).to_string(index=False))

        print(f"\n=== {name.upper()}: MONTHLY ===")
        print(monthly_metrics.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
