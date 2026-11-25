# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors

# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
build_hydro_station_mapping.py
==============================
Builds a station → bus lookup table by spatially joining each reference
station's powerplant coordinates against the clustered network buses.

Because clustering can merge several physical plants into one bus, the output
table has one row per (station, plant) pair, recording the matched bus ID.
Downstream scripts deduplicate on bus_id before summing generation so no bus
is counted twice for the same station.
"""

import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
from scipy.spatial import KDTree
from helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def parse_project_ids(value) -> set[str]:
    """Return source-qualified IDs from powerplantmatching projectID metadata."""
    if value is None:
        return set()

    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() == "nan":
            return set()
        value = ast.literal_eval(value)

    if not isinstance(value, dict):
        return set()

    qualified = set()

    for source, identifiers in value.items():
        if not isinstance(identifiers, (set, list, tuple)):
            identifiers = [identifiers]

        for identifier in identifiers:
            if pd.isna(identifier):
                continue

            identifier = str(identifier).strip()
            if identifier:
                qualified.add(f"{source}:{identifier}")

    return qualified


def build_component_mapping(
    n: pypsa.Network,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Map reference stations to individual or aggregated hydro components."""
    records = []

    for row in metadata.itertuples(index=False):
        project_ids = str(row.project_ids).split(";")

        turbines = [
            f"{project_id.strip()} turbine"
            for project_id in project_ids
            if f"{project_id.strip()} turbine" in n.links.index
        ]

        if len(turbines) > 1:
            raise ValueError(
                f"Ambiguous cascade turbine mapping for {row.station_name}"
            )

        if turbines:
            name = turbines[0]
            component = n.links.loc[name]

            if component.carrier != "hydro":
                raise ValueError(f"Unexpected carrier for {name}")

            component_type = "Link"
            station_name = row.station_name
            bus = component.bus1
            capacity = component.p_nom * component.efficiency

        else:
            name = row.fallback_component_name
            component_type = row.fallback_component_type
            station_name = row.fallback_group

            if component_type == "StorageUnit":
                table = n.storage_units
                expected_carrier = "hydro"
            elif component_type == "Generator":
                table = n.generators
                expected_carrier = "ror"
            else:
                raise ValueError(
                    f"Unsupported component type: {component_type}"
                )

            if name not in table.index:
                raise ValueError(
                    f"Hydro component {name} not found. "
                    "Check the mapping for this clustered network."
                )

            component = table.loc[name]

            if component.carrier != expected_carrier:
                raise ValueError(f"Unexpected carrier for {name}")

            bus = component.bus
            capacity = component.p_nom

        records.append({
            "station_name": station_name,
            "reference_station_name": row.station_name,
            "project_ids": row.project_ids,
            "component_type": component_type,
            "component_name": name,
            "bus_id": bus,
            "expected_capacity_mw": row.expected_capacity_mw,
            "component_capacity_mw": capacity,
        })

    mapping = pd.DataFrame(records)

    # Validate aggregated capacity and prevent cross-station double counting.
    for (_, name), group in mapping.groupby(
        ["component_type", "component_name"]
    ):
        expected = group["expected_capacity_mw"].sum()
        actual = group["component_capacity_mw"].iloc[0]

        if group["station_name"].nunique() != 1:
            raise ValueError(
                f"Component {name} is assigned to multiple station groups."
            )

        if not np.isclose(expected, actual, rtol=1e-6, atol=1e-6):
            raise ValueError(
                f"Capacity mismatch for {name}: "
                f"expected {expected} MW, found {actual} MW."
            )

    return mapping


def build_mapping(
    powerplants_path: Path,
    network_path: Path,
    station_metadata_path: Path,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns:
        station_name | plant_name | bus_id | plant_lat | plant_lon
                     | bus_lat   | bus_lon | distance_km
    """
    n = pypsa.Network(str(network_path))
    station_metadata = pd.read_csv(station_metadata_path)

    if "fallback_component_name" in station_metadata.columns:
        return build_component_mapping(n, station_metadata)

    ppl = pd.read_csv(powerplants_path)

    buses = n.buses[["x", "y"]].rename(columns={"x": "bus_lon", "y": "bus_lat"})
    buses = buses.dropna(subset=["bus_lat", "bus_lon"])
    if buses.empty:
        raise ValueError("No buses with valid coordinates found in the network.")
    kd = KDTree(buses[["bus_lat", "bus_lon"]].values)

    station_metadata = pd.read_csv(station_metadata_path)

    required = {"station_name", "project_ids"}
    missing_columns = required - set(station_metadata.columns)
    if missing_columns:
        raise ValueError(
            "Hydro station metadata is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    requested = station_metadata[["station_name", "project_ids"]].copy()
    requested["project_id"] = requested["project_ids"].str.split(";")
    requested = requested.explode("project_id")
    requested["project_id"] = requested["project_id"].str.strip()

    project_id_column = next(
        (column for column in ["projectID", "projectid"] if column in ppl.columns),
        None,
    )
    if project_id_column is None:
        raise ValueError("powerplants.csv does not contain projectID metadata.")

    plant_records = []

    for plant_index, row in ppl.iterrows():
        for project_id in parse_project_ids(row[project_id_column]):
            plant_records.append(
                {
                    "project_id": project_id,
                    "plant_index": plant_index,
                    "plant_name": row["Name"],
                    "plant_lat": row["lat"],
                    "plant_lon": row["lon"],
                }
            )

    project_lookup = pd.DataFrame(plant_records)

    df = requested.merge(project_lookup, on="project_id", how="left")

    missing = df[df["plant_index"].isna()]
    if not missing.empty:
        raise ValueError(
            "Reference station project IDs were not found in powerplants.csv:\n"
            + missing[["station_name", "project_id"]].to_string(index=False)
        )

    df = df.dropna(subset=["plant_lat", "plant_lon"])

    if df.empty:
        raise ValueError("No hydro reference stations could be mapped.")

    # Batch spatial query — all plants in one KDTree call
    dists, idxs = kd.query(df[["plant_lat", "plant_lon"]].values)

    df["bus_id"] = buses.index[idxs]
    df["bus_lat"] = buses["bus_lat"].values[idxs]
    df["bus_lon"] = buses["bus_lon"].values[idxs]
    df["distance_km"] = np.round(dists * 111.0, 2)

    far = df[df["distance_km"] > 50]
    if not far.empty:
        logger.warning(
            "The following plants mapped to a bus >50 km away — "
            "verify clustering is correct:\n%s",
            far[["station_name", "plant_name", "bus_id", "distance_km"]].to_string(),
        )

    logger.info("Mapped %d plants across %d stations:\n%s", len(df),
                df["station_name"].nunique(),
                df[["station_name", "plant_name", "bus_id", "distance_km"]].to_string(index=False))

    return df[
        [
            "station_name",
            "project_id",
            "plant_name",
            "bus_id",
            "plant_lat",
            "plant_lon",
            "bus_lat",
            "bus_lon",
            "distance_km",
        ]
    ]


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("build_hydro_station_mapping")

    configure_logging(snakemake)

    mapping = build_mapping(
        powerplants_path=Path(snakemake.input.powerplants),
        network_path=Path(snakemake.input.network),
        station_metadata_path=Path(snakemake.input.station_metadata),
    )

    out_path = Path(snakemake.output.mapping)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(out_path, index=False)
    logger.info("Saved hydro station mapping → %s", out_path)
