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


def read_busmap(path: Path) -> pd.Series:
    """Read a PyPSA-Earth busmap while preserving bus identifiers."""
    busmap = pd.read_csv(path, index_col=0, dtype=str).iloc[:, 0]
    busmap.index = busmap.index.astype(str)

    if not busmap.index.is_unique:
        raise ValueError(f"Duplicate bus identifiers in {path}")

    return busmap


def build_component_mapping(
    n: pypsa.Network,
    metadata: pd.DataFrame,
    powerplants_path: Path,
    simplify_busmap_path: Path,
    cluster_busmap_path: Path,
) -> pd.DataFrame:
    """Map reference stations to hydro components independently of clustering."""
    plants = pd.read_csv(powerplants_path, dtype={"bus": str})
    plants = plants.loc[plants["Fueltype"].eq("Hydro")].copy()

    simplify = read_busmap(simplify_busmap_path)
    cluster = read_busmap(cluster_busmap_path)

    plants["clustered_bus"] = plants["bus"].map(simplify).map(cluster)
    plants["qualified_ids"] = plants["projectID"].map(parse_project_ids)

    records = []
    assigned_plants = set()

    for row in metadata.itertuples(index=False):
        project_ids = set(map(str.strip, row.project_ids.split(";")))

        matched = plants.loc[
            plants["qualified_ids"].map(
                lambda ids: bool(ids & project_ids)
            )
        ]

        if matched.empty:
            raise ValueError(
                f"No physical plants found for {row.station_name}"
            )

        if not np.isclose(
            matched["Capacity"].sum(), row.expected_capacity_mw
        ):
            raise ValueError(
                f"Physical capacity mismatch for {row.station_name}"
            )

        for plant_index, plant in matched.iterrows():
            if plant_index in assigned_plants:
                raise ValueError(
                    f"Plant {plant['Name']} assigned to multiple stations"
                )

            assigned_plants.add(plant_index)
            bus = plant["clustered_bus"]

            if pd.isna(bus) or bus not in n.buses.index:
                raise ValueError(
                    f"No valid clustered bus for {plant['Name']}"
                )

            turbines = [
                f"{pid} turbine"
                for pid in plant["qualified_ids"]
                if f"{pid} turbine" in n.links.index
            ]

            if turbines:
                if len(turbines) != 1:
                    raise ValueError(
                        f"Ambiguous turbine mapping for {plant['Name']}"
                    )

                name = turbines[0]
                component_type = "Link"
                component = n.links.loc[name]

                if component.carrier != "hydro" or component.bus1 != bus:
                    raise ValueError(f"Invalid turbine mapping for {name}")

                capacity = component.p_nom * component.efficiency

            else:
                technology = plant["Technology"]

                if technology == "Reservoir":
                    component_type = "StorageUnit"
                    table = n.storage_units
                    carrier = "hydro"
                elif technology == "Run-Of-River":
                    component_type = "Generator"
                    table = n.generators
                    carrier = "ror"
                else:
                    raise ValueError(
                        f"Unsupported hydro technology: {technology}"
                    )

                candidates = table.loc[
                    table["bus"].eq(bus)
                    & table["carrier"].eq(carrier)
                ]

                if len(candidates) != 1:
                    raise ValueError(
                        f"Expected one {component_type} for "
                        f"{plant['Name']} at {bus}, found {len(candidates)}"
                    )

                name = candidates.index[0]
                capacity = candidates.iloc[0].p_nom

            records.append({
                "reference_station_name": row.station_name,
                "project_ids": row.project_ids,
                "plant_name": plant["Name"],
                "original_bus": plant["bus"],
                "component_type": component_type,
                "component_name": name,
                "bus_id": bus,
                "expected_capacity_mw": plant["Capacity"],
                "component_capacity_mw": capacity,
            })

    mapping = pd.DataFrame(records)

    # Merge reference stations sharing model components, including
    # transitive connections across multiple components.
    groups = [{name} for name in metadata["station_name"]]

    component_keys = ["component_type", "component_name"]

    for _, rows in mapping.groupby(component_keys):
        stations = set(rows["reference_station_name"])
        connected = [group for group in groups if group & stations]

        for group in connected:
            groups.remove(group)
            stations.update(group)

        groups.append(stations)

    station_order = metadata["station_name"].tolist()

    labels = {
        station: " + ".join(name for name in station_order if name in group)
        for group in groups
        for station in group
    }

    mapping["station_name"] = mapping["reference_station_name"].map(labels)
    mapping["comparable"] = True

    reference_capacity = metadata.set_index("station_name")[
        "expected_capacity_mw"
    ]

    mapping["reference_capacity_mw"] = mapping[
        "reference_station_name"
    ].map(reference_capacity)

    # Exclude entire station groups if their generation cannot be
    # isolated from additional or missing model capacity.
    for (_, name), rows in mapping.groupby(component_keys):
        expected = rows["expected_capacity_mw"].sum()
        actual = rows["component_capacity_mw"].iloc[0]

        if not np.isclose(expected, actual, rtol=1e-6, atol=1e-6):
            station_group = rows["station_name"].iloc[0]

            logger.warning(
                "Excluding %s: component %s has %.3f MW, "
                "but mapped reference plants account for %.3f MW.",
                station_group, name, actual, expected,
            )

            mapping.loc[
                mapping["station_name"].eq(station_group),
                "comparable",
            ] = False

    return mapping


def build_mapping(
    powerplants_path: Path,
    network_path: Path,
    station_metadata_path: Path,
    simplify_busmap_path: Path,
    cluster_busmap_path: Path,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns:
        station_name | plant_name | bus_id | plant_lat | plant_lon
                     | bus_lat   | bus_lon | distance_km
    """
    n = pypsa.Network(str(network_path))
    station_metadata = pd.read_csv(station_metadata_path)

    if "expected_capacity_mw" in station_metadata.columns:
        return build_component_mapping(
            n,
            station_metadata,
            powerplants_path,
            simplify_busmap_path,
            cluster_busmap_path,
        )

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
        simplify_busmap_path=Path(snakemake.input.simplify_busmap),
        cluster_busmap_path=Path(snakemake.input.cluster_busmap),
    )

    out_path = Path(snakemake.output.mapping)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(out_path, index=False)
    logger.info("Saved hydro station mapping → %s", out_path)
