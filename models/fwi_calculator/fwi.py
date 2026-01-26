import logging
import numpy as np
import requests
import zipfile
import datetime
from destinepyauth import get_token
from tqdm import tqdm
import xarray as xr
import matplotlib.pyplot as plt
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
            if 'bbox' in feature:
                log.info(f"BBox: {feature.get('bbox')}")
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


def _process_and_plot_fwi(h5_file: str, bbox: list, out_path: Path) -> None:
    """Read HDF5, extract FWI, crop to bbox, and plot with country borders"""
    # Create plot
    base_name = f"{Path(h5_file).name}_FWI.png"
    output_file = out_path / base_name

    # read and prepare dataset
    ds = xr.open_dataset(h5_file)
    fwi = xr.where(ds.FWI == -8000, np.nan, ds.FWI)
    fwi = xr.DataArray(np.flipud(fwi), dims=fwi.dims)
    fwi = xr.where(fwi < 0, np.nan, fwi)

    # plot
    fwi.plot(vmin = 0,cbar_kwargs = {'label':''})
    plt.title('Fire Weather Index')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    log.info(f"Plot saved: {output_file}")
    plt.close()


def main(user: str, password: str):

    os.environ['DESPAUTH_USER'] = user
    os.environ['DESPAUTH_PASSWORD'] = password

    # search past 24 hours
    dt_now = datetime.datetime.now(datetime.timezone.utc)
    # dt_now = datetime.datetime(2025, 8, 5)
    dt_yesterday = dt_now - datetime.timedelta(days=1)
    dt_now_str = dt_now.strftime(DT_FORMAT)
    dt_yesterday_str = dt_yesterday.strftime(DT_FORMAT)
    datetime_range = f"{dt_yesterday_str}/{dt_now_str}"

    # Create timestamped output directory
    out_parent = Path(".data")
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out_path = out_parent / run_timestamp
    out_path.mkdir(exist_ok=True, parents=True)
    log.info(f"Output directory: {out_path}")

    # Define Area of Interest (bbox: [west, south, east, north])
    # bbox = [2.10, 42.65, 3.25, 43.35]  # Corbieres massif
    bbox = [14, 40, 18, 42]  # southern Italy

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
        log.info("No products found for the given criteria")
        raise

    for product in results['features']:
        zip_file = _download_product(product, auth_headers, out_path)
        
        # Extract zip file
        h5_file = _extract_zip(zip_file, out_path)
        
        # Process and plot FWI
        _process_and_plot_fwi(h5_file, bbox, out_path)
        
        break


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
