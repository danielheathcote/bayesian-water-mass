import os
import glob
import csv
import numpy as np

def load_em_matrix(data_dir="data", pattern="*_EMs.npy"):
    """
    Load all .npy endmember files in data_dir matching pattern and stack them as columns.
    Returns (array, names): array is (7, n_water_masses), one row per tracer
    (potential temperature, absolute salinity, oxygen, nitrate, phosphate,
    silicate, alkalinity), and names reflect the column order.
    """
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files match {pattern} in {data_dir}")
    arrays = []
    names = []
    for f in files:
        arr = np.load(f)
        arr = np.asarray(arr).ravel()
        if arr.size != 7:
            raise ValueError(f"{f} contains {arr.size} elements; expected 7")
        arrays.append(arr)
        base = os.path.basename(f)
        names.append(base.split("_")[0])
    matrix = np.column_stack(arrays)
    return matrix, names


def get_measurement_locations(csv_path="data/SO_data_4_dan.csv"):
    """
    Load latitude/longitude columns from a CSV and return (lats, lons).
    """
    lats = []
    lons = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if "latitude" not in reader.fieldnames or "longitude" not in reader.fieldnames:
            raise ValueError("CSV must contain 'latitude' and 'longitude' columns")
        for row in reader:
            lats.append(float(row["latitude"]))
            lons.append(float(row["longitude"]))
    return lats, lons


def restrict_to_domain(lats, lons, lat_min, lat_max, lon_min, lon_max):
    """
    Keep only the measurement locations inside the rectangular model domain.

    Args:
        lats, lons: Measurement locations.
        lat_min, lat_max, lon_min, lon_max: Domain bounds (inclusive).

    Returns:
        (lats, lons) restricted to the domain.
    """
    kept = [
        (lat, lon)
        for lat, lon in zip(lats, lons)
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
    ]
    assert kept, "No measurement locations fall inside the model domain"
    lats_in_domain = [lat for lat, _ in kept]
    lons_in_domain = [lon for _, lon in kept]
    return lats_in_domain, lons_in_domain
