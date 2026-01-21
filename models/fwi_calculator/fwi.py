import requests
import os
import zipfile
import datetime
from destinepyauth import get_token
from tqdm import tqdm
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
from pathlib import Path

HDA_STAC_ENDPOINT="https://hda.data.destination-earth.eu/stac/v2"
COLLECTION_ID = "EO.EUM.DAT.MSG.LSA-FRM"
DT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _get_auth_headers():
    access_token = get_token("hda").access_token
    return {"Authorization": f"Bearer {access_token}"}


def _download_product(product: dict, auth_headers: dict) -> None:
    print(f"\n=== Downloading: {product['id']} ===")

    # Get download URL from assets
    if 'downloadLink' not in product.get('assets', {}):
        print("No downloadLink asset found")
        raise

    download_url = product['assets']['downloadLink']['href']
    filename = f"{product['id']}.zip"

    print(f"Downloading full product to: {filename}")
    print(f"URL: {download_url}")

    # Stream download with progress bar
    response = requests.get(download_url, headers=auth_headers, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as progress_bar:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

    print(f"\nDownload complete: {filename}")
    return filename


def _print_search_results(response: requests.Response) -> None:
    # print result info
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        results = response.json()
        print(f"\nFound {len(results.get('features', []))} products")
    
        # Display results
        for idx, feature in enumerate(results.get('features', [])):
            print(f"\n--- Product {idx + 1} ---")
            print(f"ID: {feature.get('id')}")
            print(f"Datetime: {feature.get('properties', {}).get('datetime')}")
            if 'bbox' in feature:
                print(f"BBox: {feature.get('bbox')}")
            print(f"Assets: {list(feature.get('assets', {}).keys())}")
    else:
        print(f"Error: {response.text}")
        raise


def _extract_zip(zip_filename: str) -> str:
    """Extract zip file and return the main HDF5 file path"""
    print(f"\n=== Extracting {zip_filename} ===")
    
    extract_dir = Path(zip_filename).stem
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(path=extract_dir)
    
    # Find HDF5 file
    h5_files = glob.glob(os.path.join(extract_dir, '*.h5')) + \
               glob.glob(os.path.join(extract_dir, '*.hdf5')) + \
               glob.glob(os.path.join(extract_dir, 'S-LSA*'))
    
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 file found in {extract_dir}")
    
    h5_file = h5_files[0]
    print(f"Found data file: {h5_file}")
    return h5_file


def _process_and_plot_fwi(h5_file: str, bbox: list) -> None:
    """Read HDF5, extract FWI, crop to bbox, and plot with country borders"""
    print(f"\n=== Processing FWI data ===")
    
    # Read the dataset
    ds = xr.open_dataset(h5_file, engine='h5netcdf')
    print(f"Dataset variables: {list(ds.data_vars)}")
    
    # Extract FWI variable (adjust variable name if needed)
    fwi_var_names = ['FWI', 'fwi', 'FireWeatherIndex']
    fwi = None
    
    for var_name in fwi_var_names:
        if var_name in ds:
            fwi = ds[var_name]
            print(f"Found FWI variable: {var_name}")
            break
    
    if fwi is None:
        print(f"Available variables: {list(ds.data_vars)}")
        # Try to use the first variable as fallback
        if len(ds.data_vars) > 0:
            fwi = ds[list(ds.data_vars)[0]]
            print(f"Using first variable as FWI: {list(ds.data_vars)[0]}")
        else:
            raise ValueError("No suitable FWI variable found")
    
    # Crop to bbox [west, south, east, north]
    west, south, east, north = bbox
    
    # Get coordinate names (try common variations)
    lon_names = ['lon', 'longitude', 'x']
    lat_names = ['lat', 'latitude', 'y']
    
    lon_dim = next((name for name in lon_names if name in fwi.coords), None)
    lat_dim = next((name for name in lat_names if name in fwi.coords), None)
    
    if lon_dim and lat_dim:
        fwi_cropped = fwi.sel(**{lon_dim: slice(west, east), lat_dim: slice(south, north)})
    else:
        print(f"Warning: Could not find lon/lat coordinates. Available coords: {list(fwi.coords)}")
        fwi_cropped = fwi
    
    # Create plot
    output_file = h5_file.replace('.h5', '_FWI.png').replace('.hdf5', '_FWI.png')
    if not output_file.endswith('.png'):
        output_file = f"{h5_file}_FWI.png"
    
    print(f"Creating plot: {output_file}")
    
    _ = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Plot FWI data
    if lon_dim and lat_dim:
        _ = fwi_cropped.plot(ax=ax, transform=ccrs.PlateCarree(), 
                              cmap='YlOrRd', cbar_kwargs={'label': 'Fire Weather Index'})
    else:
        _ = fwi_cropped.plot(ax=ax, cmap='YlOrRd', cbar_kwargs={'label': 'Fire Weather Index'})
    
    # Add country borders and coastlines
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    
    # Set extent to bbox
    ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    plt.title(f'Fire Weather Index\n{os.path.basename(h5_file)}')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {output_file}")
    plt.close()
    
    ds.close()


def main():
    # search past 24 hours
    # dt_now = datetime.datetime.now(datetime.timezone.utc)
    dt_now = datetime.datetime(2025, 8, 5)
    dt_yesterday = dt_now - datetime.timedelta(days=1)
    dt_now_str = dt_now.strftime(DT_FORMAT)
    dt_yesterday_str = dt_yesterday.strftime(DT_FORMAT)
    datetime_range = f"{dt_yesterday_str}/{dt_now_str}"

    # Define Area of Interest (bbox: [west, south, east, north])
    bbox = [2.10, 42.65, 3.25, 43.35]

    auth_headers = _get_auth_headers()

    # Search STAC catalog with filters
    response = requests.post(HDA_STAC_ENDPOINT+"/search", headers=auth_headers, json={
        "collections": [COLLECTION_ID],
        "datetime": datetime_range,
        "bbox": bbox,
        "limit": 10  # Limit number of results
    })

    _print_search_results(response)

    results = response.json()
    if len(results.get('features', [])) == 0:
        print("No products found for the given criteria")
        raise

    for product in results['features']:
        zip_file = _download_product(product, auth_headers)
        
        # Extract zip file
        h5_file = _extract_zip(zip_file)
        
        # file = "/home/dp/deltatwin/extremesdt_fwi_usecase/S-LSA_-HDF5_LSASAF_MSG_FRM-F024_Euro_202601201200/S-LSA_-HDF5_LSASAF_MSG_FRM-F024_Euro_202601201200"

        # Process and plot FWI
        _process_and_plot_fwi(h5_file, bbox)
        
        break


if __name__ == "__main__":
    main()
