import logging
import numpy as np
import requests
import zipfile
import datetime
from destinepyauth import get_token
from tqdm import tqdm
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from pathlib import Path
import os
import sys

HDA_STAC_ENDPOINT="https://hda.data.destination-earth.eu/stac/v2"
COLLECTION_ID = "EO.EUM.DAT.MSG.LSA-FRM"
DT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

logging.getLogger(__name__).addHandler(logging.NullHandler())
log = logging.getLogger("FWI")
log.setLevel(logging.INFO)

def _get_auth_headers():
    access_token = get_token("hda").access_token
    return {"Authorization": f"Bearer {access_token}"}


def _download_product(product: dict, auth_headers: dict, out_path: Path) -> None:
    log.info(f"\n=== Downloading: {product['id']} ===")

    # Get download URL from assets
    if 'downloadLink' not in product.get('assets', {}):
        log.info("No downloadLink asset found")
        raise

    download_url = product['assets']['downloadLink']['href']
    filename = out_path / f"{product['id']}.zip"

    log.info(f"Downloading full product to: {filename}")
    log.info(f"URL: {download_url}")

    # Stream download with progress bar
    response = requests.get(download_url, headers=auth_headers, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with tqdm(total=total_size, unit='B', unit_scale=True, desc=str(filename)) as progress_bar:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

    log.info(f"\nDownload complete: {filename}")
    return filename


def _print_search_results(response: requests.Response) -> None:
    # print result info
    log.info(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        results = response.json()
        log.info(f"\nFound {len(results.get('features', []))} products")
    
        # Display results
        for idx, feature in enumerate(results.get('features', [])):
            log.info(f"\n--- Product {idx + 1} ---")
            log.info(f"ID: {feature.get('id')}")
            log.info(f"Datetime: {feature.get('properties', {}).get('datetime')}")
            log.info(f"Assets: {list(feature.get('assets', {}).keys())}")
    else:
        log.info(f"Error: {response.text}")
        raise


def _extract_zip(zip_filename: str, out_path: Path) -> str:
    """Extract zip file and return the main HDF5 file path"""
    log.info(f"\n=== Extracting {zip_filename} ===")
    
    extract_dir = out_path / Path(zip_filename).stem
    extract_dir.mkdir(exist_ok=True, parents=True)
    
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(path=extract_dir)
    
    # Find HDF5 file
    h5_files = list(extract_dir.glob('*.h5')) + \
               list(extract_dir.glob('*.hdf5')) + \
               list(extract_dir.glob('S-LSA*'))
    
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 file found in {extract_dir}")
    
    h5_file = h5_files[0]
    log.info(f"Found data file: {h5_file}")
    return h5_file


def _process_and_plot_fwi(h5_file: Path, plot_index: int) -> None:
    """Read HDF5, extract FWI and plot with country borders"""
    # Create plot
    output_file = f"fwi{plot_index}.png"

    # Discrete Risk classes (1..5): green, yellow, then darker reds
    risk_colors = [
        "#2ca25f",  # 1 Low (green)
        "#ffeb3b",  # 2 Moderate (yellow)
        "#fb6a4a",  # 3 High (light red)
        "#de2d26",  # 4 Very High (red)
        "#a50f15",  # 5 Extreme (dark red)
    ]
    risk_cmap = ListedColormap(risk_colors, name="risk_classes")
    risk_bounds = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    risk_norm = BoundaryNorm(risk_bounds, risk_cmap.N)

    # read and prepare dataset
    ds = xr.open_dataset(h5_file)
    fwi = xr.where(ds.FWI == -8000, np.nan, ds.FWI)
    fwi = xr.DataArray(np.flipud(fwi), dims=fwi.dims)
    fwi = xr.where(fwi < 0, np.nan, fwi)
    # risk
    risk = xr.where(ds.Risk == -8000, np.nan, ds.Risk)
    risk = xr.DataArray(np.flipud(risk), dims=risk.dims)
    risk = xr.where(risk < 0, np.nan, risk)

    # plot
    _, axes = plt.subplots(2, 1, figsize=(8, 10))

    fwi.plot(ax=axes[0], vmin=0, cmap="Oranges", cbar_kwargs={'label': ''})
    forecast_id = str(h5_file).split("_")[-3]
    axes[0].set_title(f'Forecast {forecast_id}: Fire Weather Index')

    risk_plot = risk.plot(
        ax=axes[1],
        cmap=risk_cmap,
        norm=risk_norm,
        cbar_kwargs={
            'label': '',
            'ticks': [1, 2, 3, 4, 5],
            'boundaries': risk_bounds,
            'spacing': 'proportional',
            'drawedges': True,
        },
    )
    risk_plot.colorbar.set_ticklabels(['Low', 'Moderate', 'High', 'Very High', 'Extreme'])
    axes[1].set_title('Risk')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    log.info(f"Plot saved: {output_file}")
    plt.close()


def main(user: str, password: str, out_path: Path = Path("./.delta")):

    os.environ['DESPAUTH_USER'] = user
    os.environ['DESPAUTH_PASSWORD'] = password

    # search past 24 hours
    dt_now = datetime.datetime.now(datetime.timezone.utc)
    dt_yesterday = dt_now - datetime.timedelta(days=1)
    dt_now_str = dt_now.strftime(DT_FORMAT)
    dt_yesterday_str = dt_yesterday.strftime(DT_FORMAT)
    datetime_range = f"{dt_yesterday_str}/{dt_now_str}"

    # Create timestamped output directory
    out_path.mkdir(exist_ok=True, parents=True)
    log.info(f"Output directory: {out_path}")

    auth_headers = _get_auth_headers()

    # Search STAC catalog with filters
    response = requests.post(HDA_STAC_ENDPOINT+"/search", headers=auth_headers, json={
        "collections": [COLLECTION_ID],
        "datetime": datetime_range,
        "limit": 10  # Limit number of results
    })

    _print_search_results(response)

    results = response.json()
    if len(results.get('features', [])) == 0:
        log.info("No products found for the given criteria")
        raise

    if len(results.get('features', [])) != 5:
        log.info("Expected 5 products for 5-day forecast")
        raise

    for i, product in enumerate(results['features']):
        zip_file = _download_product(product, auth_headers, out_path)
        
        # Extract zip file
        h5_file = _extract_zip(zip_file, out_path)
        
        # Process and plot FWI
        _process_and_plot_fwi(h5_file, i)
        

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if len(sys.argv) != 3:
        log.info("Usage: python fwi.py <username> <password>")
        sys.exit(1)
    
    username = sys.argv[1]
    password = sys.argv[2]
    try:
        main(username, password)
    except Exception as e:
        log.info(f"Failed to run fwi.py:\n {e}")
