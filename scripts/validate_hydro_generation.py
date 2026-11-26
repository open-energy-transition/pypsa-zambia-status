# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors

# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
validate_hydro_generation.py
============================
Compares modelled hydro generation against ZESCO reference data at station
level for both annual (2023-2024) and monthly (Jan-Jun 2025) resolution.

Generation sources handled
--------------------------
- Reservoir hydro  → n.storage_units  (carrier "hydro")
- Run-of-river     → n.generators     (carrier "ror")

Monthly comparison is only attempted when the model's snapshot range overlaps
with the reference period (Jan-Jun 2025). If there is no overlap the rule
still succeeds, writing model-only monthly output and noting the gap.
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def load_annual_reference(path: Path) -> pd.DataFrame:
    """
    Returns DataFrame indexed by station_name with columns 2023, 2024 (GWh).
    Drops the Grand Total row automatically.
    """
    df = pd.read_csv(path)
    df.columns = ["station_name", "gen_2023_gwh", "gen_2024_gwh", "pct_change"]
    df = df[~df["station_name"].str.lower().str.contains("total", na=False)].copy()

    for col in ["gen_2023_gwh", "gen_2024_gwh"]:
        df[col] = (
            df[col].astype(str).str.replace(",", "", regex=False).astype(float)
        )

    return df.set_index("station_name")


def load_monthly_reference(path: Path) -> pd.DataFrame:
    """
    Returns DataFrame indexed by station_name with columns Jan…Jun (GWh).
    Drops the Total row automatically.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "station_name"})
    df = df[~df["station_name"].str.lower().str.contains("total", na=False)].copy()

    # Normalise station name capitalisation to match annual reference
    df["station_name"] = df["station_name"].str.strip()
    df["station_name"] = df["station_name"].replace(
        {"Victoria falls": "Victoria Falls"}
    )

    month_cols = [c for c in df.columns if c in MONTH_ABBR]
    for col in month_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["station_name"] + month_cols].set_index("station_name")


def _weighted_generation_by_bus(n: pypsa.Network) -> pd.DataFrame:
    """
    Return snapshot-weighted energy (MW·h) as DataFrame(snapshots x bus_id),
    combining reservoir (carrier='hydro') and run-of-river (carrier='ror').
    Duplicate bus columns (from clustering) are summed together.
    """
    parts = []

    hydro_sus = n.storage_units.query("carrier == 'hydro'").index
    if len(hydro_sus):
        weighted = n.storage_units_t.p[hydro_sus].mul(
            n.snapshot_weightings.generators, axis=0
        )
        weighted.columns = n.storage_units.loc[hydro_sus, "bus"]
        parts.append(weighted)
    else:
        logger.warning("No storage units with carrier='hydro' found in network.")

    ror_gens = n.generators.query("carrier == 'ror'").index
    if len(ror_gens):
        weighted = n.generators_t.p[ror_gens].mul(
            n.snapshot_weightings.generators, axis=0
        )
        weighted.columns = n.generators.loc[ror_gens, "bus"]
        parts.append(weighted)
    else:
        logger.warning("No generators with carrier='ror' found in network.")

    if not parts:
        raise ValueError("Network contains neither hydro storage units nor ror generators.")

    combined = pd.concat(parts, axis=1)
    combined.columns.name = "bus_id"
    return combined.T.groupby("bus_id").sum().T


def extract_model_generation_by_bus(n: pypsa.Network) -> pd.Series:
    """Return total generation (GWh) indexed by bus_id."""
    return _weighted_generation_by_bus(n).sum() / 1e3  # MW·h → GWh


def extract_model_generation_monthly_by_bus(n: pypsa.Network) -> pd.DataFrame:
    """Return monthly generation (GWh) as DataFrame (index=month end timestamp, columns=bus_id)."""
    return _weighted_generation_by_bus(n).resample("ME").sum() / 1e3  # MW·h → GWh


def aggregate_to_stations(
    gen_by_bus: pd.Series,
    mapping: pd.DataFrame,
) -> pd.Series:
    """
    For each named generating facility in the ERB reference data, sum
    generation across its unique mapped buses.
    Deduplicating on bus_id ensures a bus is not counted twice when two
    physical plants were merged into the same clustered bus.
    """
    result = {}
    for erb_plant_name, grp in mapping.groupby("station_name"):
        unique_buses = grp["bus_id"].astype(str).unique()
        result[erb_plant_name] = gen_by_bus.reindex(unique_buses).sum()
    return pd.Series(result)


def aggregate_monthly_to_stations(
    monthly: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Monthly version: rows=timestamp, columns=ERB facility name."""
    return monthly.apply(
        lambda month: aggregate_to_stations(month, mapping),
        axis=1,
    )


def plot_annual_comparison(
    model: pd.Series,
    ref: pd.DataFrame,
    model_year: int,
    output_path: Path,
) -> None:
    ref_col = f"gen_{model_year}_gwh"
    has_ref = ref_col in ref.columns

    stations = model.index
    if has_ref:
        stations = model.index.intersection(ref.index)
        if stations.empty:
            logger.warning("No station overlap between model and reference — plotting model only.")
            has_ref = False
            stations = model.index

    x = np.arange(len(stations))
    width = 0.35 if has_ref else 0.5

    fig, ax = plt.subplots(figsize=(8, 5))

    if has_ref:
        ref_vals = ref.loc[stations, ref_col]
        model_vals = model.loc[stations]
        ax.bar(x - width / 2, ref_vals,   width, label=f"Reference ({model_year})", color="steelblue")
        ax.bar(x + width / 2, model_vals, width, label="PyPSA-Earth model", color="coral")
        for i, (r, m) in enumerate(zip(ref_vals, model_vals)):
            if r > 0:
                pct = (m - r) / r * 100
                ax.text(x[i] + width / 2, m + 30, f"{pct:+.1f}%",
                        ha="center", va="bottom", fontsize=8)
    else:
        logger.warning("No reference data for year %d — plotting model only.", model_year)
        model_vals = model.loc[stations]
        ax.bar(x, model_vals, width, label="PyPSA-Earth model (no reference)", color="coral")

    ax.set_xticks(x)
    ax.set_xticklabels(stations, rotation=15, ha="right")
    ax.set_ylabel("Generation (GWh)")
    ax.set_title(f"Annual Hydro Generation by Station — {model_year}")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved annual comparison plot → %s", output_path)


def plot_monthly_comparison(
    model_monthly: pd.DataFrame,
    ref_monthly: pd.DataFrame | None,
    output_path: Path,
) -> None:
    stations = model_monthly.columns
    n_stations = len(stations)

    fig, axes = plt.subplots(
        n_stations, 1, figsize=(10, 4 * n_stations), sharex=False
    )
    if n_stations == 1:
        axes = [axes]

    for ax, station in zip(axes, stations):
        model_s = model_monthly[station]
        ax.plot(
            model_s.index, model_s.values,
            marker="o", color="coral", label="PyPSA-Earth model",
        )

        if ref_monthly is not None and station in ref_monthly.index:
            ref_row = ref_monthly.loc[station]
            # Align by month number
            ref_months = [MONTH_ABBR.index(m) + 1 for m in ref_row.index]
            model_months = model_s.index.month.tolist()
            overlap = [m for m in ref_months if m in model_months]

            if overlap:
                ref_ts = pd.Series(
                    ref_row.values,
                    index=pd.to_datetime(
                        [f"{model_s.index[0].year}-{m:02d}-01" for m in ref_months]
                    ) + pd.offsets.MonthEnd(0),
                )
                ax.plot(
                    ref_ts.index, ref_ts.values,
                    marker="s", color="steelblue", label=f"Reference ({model_s.index[0].year})",
                )

        ax.set_title(station)
        ax.set_ylabel("Generation (GWh)")
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    fig.suptitle("Monthly Hydro Generation by Station", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved monthly comparison plot → %s", output_path)


def extract_component_generation(
    n: pypsa.Network,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Extract snapshot-weighted electrical generation by reference station."""
    weights = n.snapshot_weightings.generators

    components = mapping.drop_duplicates(
        subset=["component_type", "component_name"]
    )

    generation = {}

    for row in components.itertuples(index=False):
        name = row.component_name

        if row.component_type == "Link":
            power = -n.links_t.p1[name]
        elif row.component_type == "StorageUnit":
            power = n.storage_units_t.p_dispatch[name]
        elif row.component_type == "Generator":
            power = n.generators_t.p[name]
        else:
            raise ValueError(
                f"Unsupported component type: {row.component_type}"
            )

        energy = power.mul(weights) / 1000

        if row.station_name not in generation:
            generation[row.station_name] = energy
        else:
            generation[row.station_name] += energy

    return pd.DataFrame(generation)


def validate_component_generation(n, mapping, snakemake):
    """Validate component-level generation against dated ERB observations."""
    if "comparable" in mapping.columns:
        mapping = mapping.loc[mapping["comparable"].eq(True)].copy()

    if mapping.empty:
        raise ValueError("No comparable hydro station groups found.")

    model_year = n.snapshots[0].year

    generation = extract_component_generation(n, mapping)

    reference_mapping = mapping[
        ["reference_station_name", "station_name"]
    ].drop_duplicates()

    if reference_mapping["reference_station_name"].duplicated().any():
        raise ValueError("Ambiguous reference station mapping.")

    annual_ref = pd.read_csv(snakemake.input.ref_annual)
    annual_ref = annual_ref.loc[annual_ref["year"].eq(model_year)]

    monthly_ref = pd.read_csv(
        snakemake.input.ref_monthly,
        parse_dates=["month"],
    )
    monthly_ref = monthly_ref.loc[
        monthly_ref["month"].dt.year.eq(model_year)
    ]

    def aggregate_reference(reference, group_columns):
        merged = reference_mapping.merge(
            reference.rename(
                columns={"station_name": "reference_station_name"}
            ),
            on="reference_station_name",
            how="left",
        )

        if merged["generation_gwh"].isna().any():
            raise ValueError("Missing generation reference observations.")

        return merged.groupby(group_columns)["generation_gwh"].sum()

    annual_ref = aggregate_reference(
        annual_ref, ["station_name"]
    )

    monthly_ref = aggregate_reference(
        monthly_ref, ["month", "station_name"]
    ).unstack("station_name")

    annual = pd.DataFrame({
        "model_gwh": generation.sum(),
        "reference_gwh": annual_ref,
    })

    annual["diff_gwh"] = (
        annual["model_gwh"] - annual["reference_gwh"]
    )

    annual["diff_pct"] = (
        100 * annual["diff_gwh"]
        / annual["reference_gwh"].replace(0, np.nan)
    )

    monthly_model = generation.resample("MS").sum()

    monthly_model.index.name = "month"
    monthly_model.columns.name = "station_name"

    monthly_ref.index.name = "month"
    monthly_ref.columns.name = "station_name"

    monthly = pd.concat(
        {
            "model_gwh": monthly_model.stack(),
            "reference_gwh": monthly_ref.stack(),
        },
        axis=1,
    )

    monthly["diff_gwh"] = (
        monthly["model_gwh"] - monthly["reference_gwh"]
    )

    monthly["diff_pct"] = (
        100 * monthly["diff_gwh"]
        / monthly["reference_gwh"].replace(0, np.nan)
    )

    # Aggregate each reference station and model component only once.
    reference_capacity = (
        mapping.drop_duplicates("reference_station_name")
        .groupby("station_name")["reference_capacity_mw"]
        .sum()
    )

    model_capacity = (
        mapping.drop_duplicates(["component_type", "component_name"])
        .groupby("station_name")["component_capacity_mw"]
        .sum()
    )

    weights = n.snapshot_weightings.generators

    annual_hours = (
        pd.Timestamp(f"{model_year + 1}-01-01")
        - pd.Timestamp(f"{model_year}-01-01")
    ).total_seconds() / 3600

    annual["model_capacity_mw"] = model_capacity
    annual["reference_capacity_mw"] = reference_capacity

    annual["model_cf_pct"] = (
        1e5 * annual["model_gwh"]
        / (annual["model_capacity_mw"] * weights.sum())
    )

    annual["reference_cf_pct"] = (
        1e5 * annual["reference_gwh"]
        / (annual["reference_capacity_mw"] * annual_hours)
    )

    annual["cf_diff_pp"] = (
        annual["model_cf_pct"] - annual["reference_cf_pct"]
        if np.isclose(weights.sum(), annual_hours)
        else np.nan
    )

    stations = monthly.index.get_level_values("station_name")
    months = monthly.index.get_level_values("month")

    monthly["model_capacity_mw"] = stations.map(model_capacity)
    monthly["reference_capacity_mw"] = stations.map(reference_capacity)

    model_hours = months.map(weights.resample("MS").sum())
    reference_hours = months.days_in_month * 24

    monthly["model_cf_pct"] = (
        1e5 * monthly["model_gwh"]
        / (monthly["model_capacity_mw"] * model_hours)
    )

    monthly["reference_cf_pct"] = (
        1e5 * monthly["reference_gwh"]
        / (monthly["reference_capacity_mw"] * reference_hours)
    )

    monthly["cf_diff_pp"] = (
        monthly["model_cf_pct"] - monthly["reference_cf_pct"]
    ).where(np.isclose(model_hours, reference_hours))

    out_annual = Path(snakemake.output.annual_csv)
    out_monthly = Path(snakemake.output.monthly_csv)

    out_annual.parent.mkdir(parents=True, exist_ok=True)
    out_monthly.parent.mkdir(parents=True, exist_ok=True)

    annual.to_csv(out_annual)
    monthly.to_csv(out_monthly)

    annual_plot_ref = annual_ref.to_frame(
        name=f"gen_{model_year}_gwh"
    )

    monthly_plot_model = monthly_model.copy()
    monthly_plot_model.index += pd.offsets.MonthEnd(0)

    monthly_plot_ref = monthly_ref.T
    monthly_plot_ref.columns = [
        MONTH_ABBR[month.month - 1]
        for month in monthly_ref.index
    ]

    plot_annual_comparison(
        generation.sum(),
        annual_plot_ref,
        model_year,
        Path(snakemake.output.annual_png),
    )

    plot_monthly_comparison(
        monthly_plot_model,
        monthly_plot_ref,
        Path(snakemake.output.monthly_png),
    )

    logger.info("Annual generation comparison:\n%s", annual.to_string())


def main(snakemake) -> None:
    network_path     = Path(snakemake.input.network)
    mapping_path     = Path(snakemake.input.mapping)
    ref_annual_path  = Path(snakemake.input.ref_annual)
    ref_monthly_path = Path(snakemake.input.ref_monthly)
    out_annual_csv   = Path(snakemake.output.annual_csv)
    out_annual_png   = Path(snakemake.output.annual_png)
    out_monthly_csv  = Path(snakemake.output.monthly_csv)
    out_monthly_png  = Path(snakemake.output.monthly_png)

    logger.info("Loading network from %s", network_path)
    n = pypsa.Network(str(network_path))

    model_year = n.snapshots[0].year
    logger.info("Model year detected: %d", model_year)

    mapping = pd.read_csv(mapping_path)
    logger.info("Loaded station mapping:\n%s", mapping.to_string())

    if "component_type" in mapping.columns:
        validate_component_generation(n, mapping, snakemake)
        return

    ref_annual  = load_annual_reference(ref_annual_path)
    ref_monthly = load_monthly_reference(ref_monthly_path)

    gen_by_bus = extract_model_generation_by_bus(n)
    model_annual = aggregate_to_stations(gen_by_bus, mapping)

    ref_col = f"gen_{model_year}_gwh"
    annual_df = pd.DataFrame({"model_gwh": model_annual})
    if ref_col in ref_annual.columns:
        annual_df["reference_gwh"] = ref_annual[ref_col]
        annual_df["diff_gwh"] = annual_df["model_gwh"] - annual_df["reference_gwh"]
        annual_df["diff_pct"] = (
            annual_df["diff_gwh"] / annual_df["reference_gwh"] * 100
        ).round(1)
    else:
        logger.warning(
            "No reference data for model year %d. Available years: %s",
            model_year, [c.replace("gen_", "").replace("_gwh", "") for c in ref_annual.columns],
        )

    out_annual_csv.parent.mkdir(parents=True, exist_ok=True)
    annual_df.to_csv(out_annual_csv)
    logger.info("Saved annual comparison CSV → %s", out_annual_csv)

    plot_annual_comparison(model_annual, ref_annual, model_year, out_annual_png)

    monthly_by_bus = extract_model_generation_monthly_by_bus(n)
    model_monthly = aggregate_monthly_to_stations(monthly_by_bus, mapping)

    ref_year = 2025
    ref_months_set = set(range(1, 7))  # Jan–Jun
    model_months_set = set(model_monthly.index.year * 100 + model_monthly.index.month)
    overlap = any(
        ref_year * 100 + m in model_months_set for m in ref_months_set
    )

    ref_monthly_aligned = ref_monthly if overlap else None
    if not overlap:
        logger.info(
            "Model snapshots (%d) do not overlap Jan–Jun 2025 reference. "
            "Writing model-only monthly output.",
            model_year,
        )

    # Rename columns to month abbreviations for readability
    monthly_out = model_monthly.copy()
    monthly_out.index = [ts.strftime("%Y-%m") for ts in monthly_out.index]
    out_monthly_csv.parent.mkdir(parents=True, exist_ok=True)
    monthly_out.to_csv(out_monthly_csv)
    logger.info("Saved monthly model CSV → %s", out_monthly_csv)

    plot_monthly_comparison(model_monthly, ref_monthly_aligned, out_monthly_png)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("validate_hydro_generation")
    configure_logging(snakemake)
    main(snakemake)
