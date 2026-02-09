import logging
import numpy as np
import requests
import zipfile
import datetime
from pathlib import Path
import os
import sys
import time

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from destinepyauth import get_token
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from PIL import Image
from pyproj import CRS
from rasterio.transform import from_origin
from tqdm import tqdm
import xarray as xr
import rioxarray

HDA_STAC_ENDPOINT="https://hda.data.destination-earth.eu/stac/v2"
COLLECTION_ID = "EO.EUM.DAT.MSG.LSA-FRM"
DT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

logging.getLogger(__name__).addHandler(logging.NullHandler())
log = logging.getLogger("FWI")
log.setLevel(logging.INFO)


def _create_gif(png_files: list[Path], out_gif: Path, frame_duration_s: float = 1) -> None:
    existing = [p for p in png_files if p.exists()]
    if not existing:
        raise FileNotFoundError("No PNG frames found to build GIF")

    frames = [Image.open(p).convert("RGBA") for p in existing]
    duration_ms = int(frame_duration_s * 1000)
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    log.info(f"GIF saved: {out_gif}")


def _get_auth_headers():
    access_token = get_token("hda").access_token
    return {"Authorization": f"Bearer {access_token}"}


def _download_product(product: dict, auth_headers: dict, out_path: Path) -> Path:
    log.info(f"\n=== Downloading: {product['id']} ===")

    # Get download URL from assets
    if 'downloadLink' not in product.get('assets', {}):
        log.info("No downloadLink asset found")
        raise

    download_url = product['assets']['downloadLink']['href']
    filename = out_path / f"{product['id']}.zip"

    log.info(f"Downloading full product to: {filename}")
    log.info(f"URL: {download_url}")

    # Simple retry mechanism for transient errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Stream download with progress bar
            response = requests.get(download_url, headers=auth_headers, stream=True, timeout=30)
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
            
        except (requests.RequestException, OSError) as e:
            if attempt < max_retries - 1:
                log.warning(f"Download failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in 5s...")
                time.sleep(5)
            else:
                log.error(f"Download failed after {max_retries} attempts")
                raise


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


def _extract_zip(zip_filename: Path, out_path: Path) -> Path:
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


def _reproject_geos(ds: xr.Dataset):
    # # rename dims
    ds = ds.rename({"phony_dim_1": "x", "phony_dim_0": "y"})

    CFAC = ds.attrs["CFAC"]
    LFAC = ds.attrs["LFAC"]
    COFF = ds.attrs["COFF"]
    LOFF = ds.attrs["LOFF"]

    # Satellite height above ellipsoid (MSG)
    H = 35786023.0

    # Convert CFAC/LFAC from deg^-1 to rad^-1 (this is the missing ~57.2958 factor)
    deg_per_rad = 180.0 / np.pi
    CFAC_rad = CFAC * deg_per_rad
    LFAC_rad = LFAC * deg_per_rad

    # Scan-angle step (radians per pixel): 2^16 / CFAC_rad
    dx_ang = (2**16) / CFAC_rad
    dy_ang = (2**16) / LFAC_rad

    # Convert scan-angle step to PROJ geos meters using: proj_coord = H * scan_angle
    dx = H * dx_ang
    dy = H * dy_ang

    # Upper-left corner in projected meters
    x0 = -COFF * dx
    y0 =  LOFF * dy

    transform = from_origin(x0, y0, dx, dy)

    crs = CRS.from_proj4(
        "+proj=geos +h=35786023 +lon_0=0 "
        "+a=6378137 +b=6356752.31414 +sweep=x +units=m +no_defs"
    )

    out = {}
    for var in ["FWI", "Risk"]:

        da = ds[var].rio.set_spatial_dims("x", "y")
        da = da.rio.write_crs(crs)
        da = da.rio.write_transform(transform)

        fill = da.attrs.get("_FillValue", -32768)
        da = da.rio.write_nodata(fill)

        da = da.rio.reproject("EPSG:3035")   # Europe LAEA

        # all europe
        # bbox = {
        #     "minx": 250000,
        #     "miny": 1400000,
        #     "maxx": 7500000,
        #     "maxy": 5500000,
        # }
        # Italy
        bbox = {
            "minx": 4000000,
            "miny": 1400000,
            "maxx": 5200000,
            "maxy": 2600000,
        }

        da = da.rio.clip_box(
            minx=bbox["minx"],
            miny=bbox["miny"],
            maxx=bbox["maxx"],
            maxy=bbox["maxy"],
        )

        da = da.where(da != fill).astype("float32")
        # 1) Force 2D (drop band/time/etc. if present)
        da = da.squeeze(drop=True)

        # 2) Force dims to be exactly ("y", "x")
        # (if your dims are already y/x, this is a no-op)
        if da.dims != ("y", "x"):
            da = da.rename({da.dims[-2]: "y", da.dims[-1]: "x"})

        t = da.rio.transform()
        ny, nx = da.shape

        x = t.c + (np.arange(nx) + 0.5) * t.a      # pixel centers
        y = t.f + (np.arange(ny) + 0.5) * t.e

        da = da.assign_coords(x=("x", x), y=("y", y))

        out[var] = da
        
    return out["FWI"], out["Risk"]

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
    ds = xr.open_dataset(h5_file, engine="h5netcdf", phony_dims="sort")
    fwi, risk = _reproject_geos(ds)
    fwi = xr.where(fwi == -8000, np.nan, fwi)
    risk = xr.where(risk == -8000, np.nan, risk)

    # get forecast_id for titles
    forecast_id = str(h5_file).split("_")[-3]
    forecast_offset = forecast_id[-3:]
    forecast_dt = datetime.datetime.strptime(str(h5_file).split("_")[-1], "%Y%m%d%H%M")
    forecast_dt = forecast_dt.strftime("%Y-%m-%d, %H%M")
    forecast_descr = f"{forecast_dt} Z +{forecast_offset}h"


    # plot
    plot_crs = ccrs.epsg(3035)
    _, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": plot_crs})

    borders = cfeature.BORDERS.with_scale("50m")
    coastlines = cfeature.COASTLINE.with_scale("50m")

    fwi_cmap = plt.get_cmap("YlOrRd", 5)

    fwi.plot(
        ax=axes[0],
        vmin=0,
        vmax=5000,
        cmap=fwi_cmap,
        transform=plot_crs,
        cbar_kwargs={"label": ""},
    )
    axes[0].add_feature(borders, linewidth=1)
    axes[0].add_feature(coastlines, linewidth=1)
    axes[0].set_title(f"MSG Fire Weather Index at {forecast_descr}")

    risk_plot = risk.plot(
        ax=axes[1],
        cmap=risk_cmap,
        norm=risk_norm,
        transform=plot_crs,
        cbar_kwargs={
            'label': '',
            'ticks': [1, 2, 3, 4, 5],
            'boundaries': risk_bounds,
            'spacing': 'proportional',
            'drawedges': True,
        },
    )
    risk_plot.colorbar.set_ticklabels(['Low', 'Moderate', 'High', 'Very High', 'Extreme'])
    axes[1].add_feature(borders, linewidth=1)
    axes[1].add_feature(coastlines, linewidth=1)
    axes[1].set_title(f"MSG Fire Risk at {forecast_descr}")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    log.info(f"Plot saved: {output_file}")
    plt.close()


def main(out_path: Path = Path("./.delta")):

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

    # Build animated GIF from generated plots (0.5s per frame)
    frames = [Path(f"fwi{i}.png") for i in range(len(results["features"]))]
    _create_gif(frames, Path("fwi_forecast.gif"), frame_duration_s=1)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if len(sys.argv) not in (3, 1):
        log.info("Usage: python fwi.py <username> <password>")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        username = sys.argv[1]
        password = sys.argv[2]
        os.environ['DESPAUTH_USER'] = username
        os.environ['DESPAUTH_PASSWORD'] = password
    try:
        main()
    except Exception as e:
        log.info(f"Failed to run fwi.py:\n {e}")
