import os
import glob
import json
import gzip
import boto3
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timezone, timedelta
from botocore import UNSIGNED
from botocore.config import Config

# --- Configuration ---
FRAMES_DIR = "docs/frames"
MAX_FRAMES = 15  # Retain only the 15 newest frames per product
MAX_AGE_DAYS = 1  # Remove frames older than 1 day

# Categorical colormap for Precipitation Type (MRMS PrecipFlag values)
# Values: 0=No Precip, 1=Rain, 3=Snow, 4=Mixed/Ice Pellets,
#         6=Freezing Rain, 7=Hail, 10=Cool-Season Rain, 91=Convective Rain
PRECIP_TYPE_COLORS = {
    0:  ("#000000", "No Precip"),
    1:  ("#00c800", "Rain"),
    2:  ("#00c800", "Rain"),
    3:  ("#6464ff", "Snow"),
    4:  ("#ff9600", "Mixed / Ice Pellets"),
    6:  ("#e600e6", "Freezing Rain"),
    7:  ("#ff0000", "Hail"),
    10: ("#0064ff", "Cool-Season Rain"),
    91: ("#ffff00", "Convective Rain"),
    96: ("#ff6400", "Tropical Rain"),
    97: ("#ff0000", "Heavy Tropical Rain"),
}

def make_precip_type_cmap():
    """Build a ListedColormap and BoundaryNorm for MRMS PrecipFlag."""
    # We cover 0-10 explicitly, then a catch-all bin for values ≥11
    boundaries = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 100]
    colors_list = [
        "#000000",  # 0  No Precip
        "#00c800",  # 1  Rain
        "#00c800",  # 2  Rain (alt)
        "#6464ff",  # 3  Snow
        "#ff9600",  # 4  Mixed / Ice Pellets
        "#ff9600",  # 5  (unused – same as mixed)
        "#e600e6",  # 6  Freezing Rain
        "#ff0000",  # 7  Hail
        "#888888",  # 8  (unused)
        "#888888",  # 9  (unused)
        "#0064ff",  # 10 Cool-Season Rain
        "#ffff00",  # 91-99 → Convective / Tropical Rain (catch-all)
    ]
    cmap = mcolors.ListedColormap(colors_list)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    return cmap, norm

PRODUCTS = {
    "Reflectivity": {
        "prefix": "CONUS/MergedReflectivityQCComposite_00.50",
        "cmap_type": "pal",
        "cmap_source": "RadarScope1.pal",
        "unit": "dBZ",
        "vmin": None,
        "vmax": None,
    },
    "PrecipType": {
        "prefix": "CONUS/PrecipFlag",
        "cmap_type": "categorical",
        "cmap_source": None,
        "unit": "Precipitation Type",
        "vmin": 0,
        "vmax": 11,
        "legend": PRECIP_TYPE_COLORS,
    },
    "Rotation_1hr": {
        "prefix": "CONUS/RotationTrack60min_0-2kmAGL",
        "cmap_type": "mpl",
        "cmap_source": "RdBu_r",
        "unit": "1hr Max Rotation AGL (s\u207b\u00b9)",
        "vmin": -0.005,
        "vmax":  0.005,
    },
    "Rotation_Instant": {
        "prefix": "CONUS/MergedAzimuthalShear_0-2kmAGL",
        "cmap_type": "mpl",
        "cmap_source": "RdBu_r",
        "unit": "Instant Max Rotation AGL (s\u207b\u00b9)",
        "vmin": -0.01,
        "vmax":  0.01,
    },
    "MESH": {
        "prefix": "CONUS/MESH",
        "cmap_type": "mpl",
        "cmap_source": "plasma",
        "unit": "Max Expected Hail Size (mm)",
        "vmin": 0,
        "vmax": 100,
    },
    "PrecipRate": {
        "prefix": "CONUS/PrecipRate",
        "cmap_type": "mpl",
        "cmap_source": "YlGnBu",
        "unit": "Precipitation Rate (mm/hr)",
        "vmin": 0.1,
        "vmax": 100,
    },
    "QPE_1hr": {
        "prefix": "CONUS/RadarOnly_QPE_01H",
        "cmap_type": "mpl",
        "cmap_source": "YlGnBu",
        "unit": "1hr QPE (mm)",
        "vmin": 0.1,
        "vmax": 150,
    },
    "QPE_6hr": {
        "prefix": "CONUS/RadarOnly_QPE_06H",
        "cmap_type": "mpl",
        "cmap_source": "YlGnBu",
        "unit": "6hr QPE (mm)",
        "vmin": 0.1,
        "vmax": 150,
    },
    "QPE_24hr": {
        "prefix": "CONUS/RadarOnly_QPE_24H",
        "cmap_type": "mpl",
        "cmap_source": "YlGnBu",
        "unit": "24hr QPE (mm)",
        "vmin": 0.1,
        "vmax": 150,
    },
}


def ensure_directories():
    os.makedirs(FRAMES_DIR, exist_ok=True)


def create_colormap_from_pal(pal_file):
    colors = []
    with open(pal_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] in ['color:', 'color4:']:
                val = float(parts[1])
                r, g, b = float(parts[2]) / 255, float(parts[3]) / 255, float(parts[4]) / 255
                if parts[0] == 'color4:':
                    a = float(parts[5]) / 255 if len(parts) > 5 else 1.0
                    colors.append((val, (r, g, b, a)))
                else:
                    colors.append((val, (r, g, b)))

    min_val = min(c[0] for c in colors)
    max_val = max(c[0] for c in colors)

    normalized_colors = []
    for val, color in colors:
        norm_val = (val - min_val) / (max_val - min_val)
        normalized_colors.append((norm_val, color))

    cmap = mcolors.LinearSegmentedColormap.from_list("RadarScopeCmap", normalized_colors)
    return cmap, min_val, max_val


def download_latest_mrms(prefix, product_key):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'noaa-mrms-pds'
    now = datetime.now(timezone.utc)

    date_str = now.strftime('%Y%m%d')
    full_prefix = f"{prefix}/{date_str}/"

    response = s3.list_objects_v2(Bucket=bucket, Prefix=full_prefix)
    if 'Contents' not in response:
        print(f"No data found for {product_key} today yet.")
        return None

    files = [obj for obj in response['Contents'] if obj['Key'].endswith('.grib2.gz')]
    if not files:
        return None

    latest_file_key = sorted(files, key=lambda x: x['LastModified'])[-1]['Key']
    local_gz_path = f"latest_{product_key}.grib2.gz"
    local_grib_path = f"latest_{product_key}.grib2"

    print(f"Downloading {product_key}...")
    s3.download_file(bucket, latest_file_key, local_gz_path)

    with gzip.open(local_gz_path, 'rb') as f_in:
        with open(local_grib_path, 'wb') as f_out:
            f_out.write(f_in.read())

    return local_grib_path


def add_precip_type_legend(ax, fig):
    """Overlay a compact categorical legend for PrecipType in place of a colorbar."""
    legend_entries = [
        (color, label)
        for val, (color, label) in sorted(PRECIP_TYPE_COLORS.items())
        if val in (0, 1, 3, 4, 6, 7, 10, 91)
    ]
    patches = [
        plt.Rectangle((0, 0), 1, 1, fc=color, label=label)
        for color, label in legend_entries
    ]
    legend = ax.legend(
        handles=patches,
        loc='lower left',
        fontsize=8,
        framealpha=0.6,
        facecolor='#1a1a1a',
        edgecolor='white',
        labelcolor='white',
        ncol=2,
    )


def plot_data(grib_path, product_key, config):
    ds = xr.open_dataset(grib_path, engine='cfgrib')
    var_name = list(ds.data_vars)[0]
    data = ds[var_name]

    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, facecolor='#111111')
    ax.add_feature(cfeature.LAND, facecolor='#222222')
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, edgecolor='white', linewidth=0.5)
    ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.25)

    if config["cmap_type"] == "pal":
        cmap, vmin, vmax = create_colormap_from_pal(config["cmap_source"])
        plot_field = data.where(data >= vmin) if vmin is not None else data
        norm = None

    elif config["cmap_type"] == "categorical":
        cmap, norm = make_precip_type_cmap()
        vmin = config["vmin"]
        vmax = config["vmax"]
        plot_field = data

    else:  # standard matplotlib colormap
        cmap = plt.get_cmap(config["cmap_source"])
        vmin = config["vmin"]
        vmax = config["vmax"]
        norm = None
        plot_field = data.where(data >= vmin) if vmin is not None else data

    mesh = ax.pcolormesh(
        ds.longitude, ds.latitude, plot_field,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin if norm is None else None,
        vmax=vmax if norm is None else None,
        norm=norm,
        shading='auto',
    )

    if config["cmap_type"] == "categorical":
        add_precip_type_legend(ax, fig)
    else:
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.04, aspect=40, shrink=0.7)
        cbar.ax.xaxis.set_tick_params(color='white')
        cbar.ax.tick_params(axis='x', colors='white')
        cbar.set_label(config["unit"], color='white')

    valid_time = np.datetime_as_string(data.time.values, unit='m').replace('T', ' ') + ' UTC'
    title_text = f"MRMS {product_key.replace('_', ' ')} (CONUS) | Valid: {valid_time}"
    plt.title(title_text, color='white', loc='center', pad=15, fontsize=16, fontweight='bold')

    fig.patch.set_facecolor('#1a1a1a')

    timestamp_str = np.datetime_as_string(data.time.values, unit='s').replace('T', '_').replace(':', '')
    output_filename = f"{FRAMES_DIR}/{product_key}_{timestamp_str}.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close()

    # Clean up the large grib files to save space on the runner
    os.remove(grib_path)
    gz_path = grib_path + ".gz"
    if os.path.exists(gz_path):
        os.remove(gz_path)


def frame_timestamp(filename):
    """Parse the UTC datetime embedded in a frame filename like Product_2026-03-12_142600.png."""
    basename = os.path.basename(filename)
    # strip product prefix and .png suffix, leaving e.g. "2026-03-12_142600"
    parts = basename.rsplit('_', 2)
    if len(parts) < 3:
        return None
    try:
        date_part = parts[-2]   # "2026-03-12"
        time_part = parts[-1].replace('.png', '')  # "142600"
        dt = datetime.strptime(f"{date_part}_{time_part}", "%Y-%m-%d_%H%M%S")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def update_frame_list():
    """Manage rolling window (15 frames, max 1 day old) per product and update JSON."""
    frame_dict = {}
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)

    for product in PRODUCTS.keys():
        frames = sorted(glob.glob(f"{FRAMES_DIR}/{product}_*.png"))

        # Remove frames older than MAX_AGE_DAYS
        kept = []
        for f in frames:
            ts = frame_timestamp(f)
            if ts is None or ts < cutoff:
                print(f"Removing old frame: {os.path.basename(f)}")
                os.remove(f)
            else:
                kept.append(f)
        frames = kept

        # Trim to MAX_FRAMES, removing oldest first
        while len(frames) > MAX_FRAMES:
            print(f"Removing excess frame: {os.path.basename(frames[0])}")
            os.remove(frames[0])
            frames.pop(0)

        frame_dict[product] = [os.path.basename(f) for f in frames]

    with open("docs/frames.json", "w") as f:
        json.dump(frame_dict, f, indent=4)


if __name__ == "__main__":
    ensure_directories()

    for prod_key, config in PRODUCTS.items():
        try:
            grib_file = download_latest_mrms(config["prefix"], prod_key)
            if grib_file:
                plot_data(grib_file, prod_key, config)
        except Exception as e:
            print(f"Error processing {prod_key}: {e}")

    update_frame_list()
