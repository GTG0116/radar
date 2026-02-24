import os
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

def create_colormap_from_pal(pal_file):
    """Parses a RadarScope-style .pal file into a Matplotlib colormap."""
    colors = []
    
    with open(pal_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
                
            # Handle standard colors and the RGBA color4
            if parts[0] in ['color:', 'color4:']:
                val = float(parts[1])
                # Normalize RGB from 0-255 to 0-1
                r, g, b = float(parts[2])/255, float(parts[3])/255, float(parts[4])/255
                
                # If it's color4, it includes an alpha channel
                if parts[0] == 'color4:':
                    a = float(parts[5])/255 if len(parts) > 5 else 1.0
                    colors.append((val, (r, g, b, a)))
                else:
                    colors.append((val, (r, g, b)))

    # Normalize values between 0 and 1 for Matplotlib's LinearSegmentedColormap
    min_val = min(c[0] for c in colors)
    max_val = max(c[0] for c in colors)
    
    normalized_colors = []
    for val, color in colors:
        norm_val = (val - min_val) / (max_val - min_val)
        normalized_colors.append((norm_val, color))
        
    cmap = mcolors.LinearSegmentedColormap.from_list("RadarScopeCmap", normalized_colors)
    return cmap, min_val, max_val

def download_latest_mrms():
    """Finds and downloads the latest MRMS Reflectivity GRIB2 file from AWS S3."""
    print("Connecting to NOAA MRMS S3 bucket...")
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'noaa-mrms-pds'
    
    # MRMS path structure: CONUS/MergedReflectivityQCComposite_00.50/YYYYMMDD/
    now = datetime.now(timezone.utc)
    prefix = f"CONUS/MergedReflectivityQCComposite_00.50/{now.strftime('%Y%m%d')}/"
    
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' not in response:
        raise ValueError("No data found for today yet.")
        
    # Get the latest file ending in .grib2.gz
    files = [obj for obj in response['Contents'] if obj['Key'].endswith('.grib2.gz')]
    latest_file_key = sorted(files, key=lambda x: x['LastModified'])[-1]['Key']
    
    local_gz_path = "latest_mrms.grib2.gz"
    local_grib_path = "latest_mrms.grib2"
    
    print(f"Downloading {latest_file_key}...")
    s3.download_file(bucket, latest_file_key, local_gz_path)
    
    print("Extracting GRIB2 file...")
    with gzip.open(local_gz_path, 'rb') as f_in:
        with open(local_grib_path, 'wb') as f_out:
            f_out.write(f_in.read())
            
    return local_grib_path

def plot_data(grib_path, pal_file):
    print("Loading color table...")
    cmap, vmin, vmax = create_colormap_from_pal(pal_file)
    
    print("Opening dataset...")
    # Open the unzipped GRIB2 file
    ds = xr.open_dataset(grib_path, engine='cfgrib')
    
    # Extract the main data variable (usually 'unknown' or 'refc' in MRMS gribs)
    var_name = list(ds.data_vars)[0] 
    data = ds[var_name]
    
    print("Generating plot...")
    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.Mercator())
    
    # Set boundaries to the Continental US (CONUS)
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.OCEAN, facecolor='#111111') # Dark background
    ax.add_feature(cfeature.LAND, facecolor='#222222')
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, edgecolor='white', linewidth=0.5)
    ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.25)
    
    # Plot the radar data
    mesh = ax.pcolormesh(
        ds.longitude, ds.latitude, data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading='auto'
    )
    
    # Add Colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Reflectivity (dBZ)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.tick_params(axis='y', colors='white')

    # Formatting metadata for the title
    valid_time = np.datetime_as_string(data.time.values, unit='m').replace('T', ' ') + ' UTC'
    title_text = f"MRMS Base Reflectivity (CONUS)\nDate/Time: {valid_time} | Color Table: {pal_file} | Source: MRMS AWS S3"
    
    plt.title(title_text, color='white', loc='left', pad=10, fontsize=12)
    
    # Change figure background to dark
    fig.patch.set_facecolor('#1a1a1a')
    
    print("Saving image...")
    plt.savefig('mrms_conus_radar.png', bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    print("Done! Saved as mrms_conus_radar.png")

if __name__ == "__main__":
    grib_file = download_latest_mrms()
    plot_data(grib_file, 'RadarScope1.pal')
