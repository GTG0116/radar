import os
import glob
import json
import gzip
import boto3
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from datetime import datetime, timezone
from botocore import UNSIGNED
from botocore.config import Config

# --- Configuration ---
FRAMES_DIR = "docs/frames"
MAX_FRAMES = 10 # Retain only the 10 newest frames per product

# Define the products we want to map. 
# You can add the Rotation tracks here if the GitHub runner handles the load!
PRODUCTS = {
    "Reflectivity": {
        "prefix": "CONUS/MergedReflectivityQCComposite_00.50",
        "cmap_type": "pal",
        "cmap_source": "RadarScope1.pal",
        "unit": "dBZ",
        "vmin": None, # Handled by the pal file
        "vmax": None
    },
    "MESH": {
        "prefix": "CONUS/MESH",
        "cmap_type": "mpl",
        "cmap_source": "plasma", # Built-in Matplotlib color
        "unit": "Max Estimated Hail Size (mm)",
        "vmin": 0,
        "vmax": 100
    },
    "QPE_24hr": {
        "prefix": "CONUS/RadarOnly_QPE_24H",
        "cmap_type": "mpl",
        "cmap_source": "YlGnBu", # Built-in Matplotlib color
        "unit": "24hr Precip (mm)",
        "vmin": 0.1,
        "vmax": 150
    }
    # Future additions:
    # "Instant_Rotation": {"prefix": "CONUS/MergedAzimuthalShear_0-2kmAGL", ...}
    # "1hr_Rotation": {"prefix": "CONUS/RotationTrack60min_0-2kmAGL", ...}
    # "24hr_Rotation": {"prefix": "CONUS/RotationTrack1440min_0-2kmAGL", ...}
    # "POSH": {"prefix": "CONUS/POSH", ...}
}

def ensure_directories():
    os.makedirs(FRAMES_DIR, exist_ok=True)

def create_colormap_from_pal(pal_file):
    colors = []
    with open(pal_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if parts[0] in ['color:', 'color4:']:
                val = float(parts[1])
                r, g, b = float(parts[2])/255, float(parts[3])/255, float(parts[4])/255
                if parts[0] == 'color4:':
                    a = float(parts[5])/255 if len(parts) > 5 else 1.0
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
    
    # Check today's folder
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

def plot_data(grib_path, product_key, config):
    if config["cmap_type"] == "pal":
        cmap, vmin, vmax = create_colormap_from_pal(config["cmap_source"])
    else:
        cmap = plt.get_cmap(config["cmap_source"])
        vmin = config["vmin"]
        vmax = config["vmax"]

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
    
    # Handle values below vmin (e.g., 0 for precip/hail) to make them transparent
    plot_data = data.where(data >= vmin) if vmin is not None else data

    mesh = ax.pcolormesh(
        ds.longitude, ds.latitude, plot_data,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax, shading='auto'
    )
    
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.04, aspect=40, shrink=0.7)
    cbar.ax.xaxis.set_tick_params(color='white')
    cbar.ax.tick_params(axis='x', colors='white')
    cbar.set_label(config["unit"], color='white')
    
    valid_time = np.datetime_as_string(data.time.values, unit='m').replace('T', ' ') + ' UTC'
    title_text = f"MRMS {product_key} (CONUS) | Valid: {valid_time}"
    plt.title(title_text, color='white', loc='center', pad=15, fontsize=16, fontweight='bold')
    
    fig.patch.set_facecolor('#1a1a1a')
    
    timestamp_str = np.datetime_as_string(data.time.values, unit='s').replace('T', '_').replace(':', '')
    output_filename = f"{FRAMES_DIR}/{product_key}_{timestamp_str}.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    
    # Clean up the large grib files to save space on the runner
    os.remove(grib_path)
    os.remove(grib_path + ".gz")

def update_frame_list():
    """Manages the rolling window of frames per product and updates JSON"""
    frame_dict = {}
    
    for product in PRODUCTS.keys():
        # Find frames just for this product
        frames = sorted(glob.glob(f"{FRAMES_DIR}/{product}_*.png"))
        
        # Delete oldest frames if we exceed MAX_FRAMES (10)
        while len(frames) > MAX_FRAMES:
            os.remove(frames[0])
            frames.pop(0)
            
        frame_dict[product] = [os.path.basename(f) for f in frames]

    # Save dictionary as JSON so web viewer can access different loops
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
