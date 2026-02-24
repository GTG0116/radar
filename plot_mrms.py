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

# Configuration
FRAMES_DIR = "docs/frames"
MAX_FRAMES = 24 # Keeps the last 24 runs

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

def download_latest_mrms():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'noaa-mrms-pds'
    now = datetime.now(timezone.utc)
    prefix = f"CONUS/MergedReflectivityQCComposite_00.50/{now.strftime('%Y%m%d')}/"
    
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' not in response:
        raise ValueError("No data found for today yet.")
        
    files = [obj for obj in response['Contents'] if obj['Key'].endswith('.grib2.gz')]
    latest_file_key = sorted(files, key=lambda x: x['LastModified'])[-1]['Key']
    
    local_gz_path = "latest_mrms.grib2.gz"
    local_grib_path = "latest_mrms.grib2"
    
    s3.download_file(bucket, latest_file_key, local_gz_path)
    with gzip.open(local_gz_path, 'rb') as f_in:
        with open(local_grib_path, 'wb') as f_out:
            f_out.write(f_in.read())
            
    return local_grib_path

def plot_data(grib_path, pal_file):
    cmap, vmin, vmax = create_colormap_from_pal(pal_file)
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
    
    mesh = ax.pcolormesh(
        ds.longitude, ds.latitude, data,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax, shading='auto'
    )
    
    # Horizontal colorbar at the bottom
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.04, aspect=40, shrink=0.7)
    cbar.ax.xaxis.set_tick_params(color='white')
    cbar.ax.tick_params(axis='x', colors='white')
    
    # Simplified Title
    valid_time = np.datetime_as_string(data.time.values, unit='m').replace('T', ' ') + ' UTC'
    title_text = f"MRMS Base Reflectivity (CONUS) | Valid: {valid_time}"
    plt.title(title_text, color='white', loc='center', pad=15, fontsize=16, fontweight='bold')
    
    fig.patch.set_facecolor('#1a1a1a')
    
    # Save with timestamp
    timestamp_str = np.datetime_as_string(data.time.values, unit='s').replace('T', '_').replace(':', '')
    output_filename = f"{FRAMES_DIR}/radar_{timestamp_str}.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    
    update_frame_list()

def update_frame_list():
    """Manages the rolling window of frames and generates frames.json"""
    frames = sorted(glob.glob(f"{FRAMES_DIR}/radar_*.png"))
    
    # Delete oldest frames if we exceed MAX_FRAMES
    while len(frames) > MAX_FRAMES:
        os.remove(frames[0])
        frames.pop(0)
        
    # Create JSON array of just the filenames
    frame_filenames = [os.path.basename(f) for f in frames]
    with open("docs/frames.json", "w") as f:
        json.dump(frame_filenames, f)

if __name__ == "__main__":
    ensure_directories()
    grib_file = download_latest_mrms()
    plot_data(grib_file, 'RadarScope1.pal')
