import numpy as np
from odc import stac as odc_stac
from pyproj import CRS
import requests
import xml.etree.ElementTree as ET


degToRad = np.pi / 180

bands = ["green", "red", "rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22"]
names_res = ["B03_10m", "B04_10m", "B05_20m", "B06_20m", "B07_20m", "B8A_20m", "B11_20m", "B12_20m"]
names = ["B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
band_names = {name: bands[i] for i, name in enumerate(names)}

bands_10m = ["red", "green", "blue", "nir"]
bands_20m = ["swir22", "rededge2", "rededge3", "rededge1", "swir16", "nir08"]
bands_60m = ["coastal", "nir09"]
bands_none = ["visual", "wvp", "scl", "aot", "cloud", "snow"]



bands = ["green", "red", "rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22"]
names_res = ["B03_10m", "B04_10m", "B05_20m", "B06_20m", "B07_20m", "B8A_20m", "B11_20m", "B12_20m"]
names = ["B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
band_names = {name: bands[i] for i, name in enumerate(names)}

bands_10m = ["red", "green", "blue", "nir"]
names_10m = ["B02_10m", "B03_10m", "B04_10m", "B08_10m"]
band_names_10m = {name: bands_10m[i] for i, name in enumerate(names_10m)}

bands_20m = ["rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22"]
names_20m = ["B05_20m", "B06_20m", "B07_20m", "B8A_20m", "B11_20m", "B12_20m"]
band_names_20m = {name: bands_20m[i] for i, name in enumerate(names_20m)}

bands_60m = ["coastal", "nir09"]
names_60m = ["B01_60m", "B09_60m"]
band_names_60m = {name: bands_60m[i] for i, name in enumerate(names_60m)}

bands_none = ["wvp", "scl", "aot"]
names_none = ["WVP_10m", "SCL_20m", "AOT_10m"]
band_names_none = {name: bands_none[i] for i, name in enumerate(names_none)}


def normalize(unnormalized, min, max):
    return 2 * (unnormalized - min) / (max - min) - 1


def denormalize(normalized, min, max):
    return 0.5 * (normalized + 1) * (max - min) + min


def tansig(input):
    return 2 / (1 + np.exp(-2 * input)) - 1


def load_data(items, bands=bands, chunks={'time': -1, 'x': 1024, 'y': 1024}, resolution=10):
    if not isinstance(items, list):
        items = [items]
    item = items[0]
    
    if "proj:epsg" in item.properties:
        crs = CRS.from_epsg(item.properties["proj:epsg"])
    elif "proj:wkt" in item.properties:
        crs = CRS.from_wkt(item.properties["proj:wkt"])
    elif "proj:wkt2" in item.properties:
        crs = CRS.from_wkt(item.properties["proj:wkt2"])
    elif "proj:code" in item.properties:
        code = item.properties["proj:code"]
        if code.startswith("EPSG:"):
            code = code.split(":")[-1]
        crs = CRS.from_epsg(code)
    else:
        print("Could not find CRS from item properties: ", item.properties)
    cube = odc_stac.load(items,
        crs=crs,
        bands=bands,
        chunks=chunks,
        resolution=(resolution))
    cube = cube.transpose("time", ...)

    return cube

def get_bands(items, data=None):
    
    band_dim = "bands"
    if not data:
        data = load_data(items).to_array(dim=band_dim)

    b03 = data.sel({band_dim: band_names["B03"]})*0.0001-0.1
    b04 = data.sel({band_dim: band_names["B04"]})*0.0001-0.1
    b05 = data.sel({band_dim: band_names["B05"]})*0.0001-0.1
    b06 = data.sel({band_dim: band_names["B06"]})*0.0001-0.1
    b07 = data.sel({band_dim: band_names["B07"]})*0.0001-0.1
    b8a = data.sel({band_dim: band_names["B8A"]})*0.0001-0.1
    b11 = data.sel({band_dim: band_names["B11"]})*0.0001-0.1
    b12 = data.sel({band_dim: band_names["B12"]})*0.0001-0.1

    viewZen, viewAzim, sunZen, sunAzim = get_viewing_angles(items)

    return b03, b04, b05, b06, b07, b8a, b11, b12, viewZen, viewAzim, sunZen, sunAzim


def get_viewing_angles(items):
    vza, vaa, sza, saa = [], [], [], []
    for item in items:
        if "view:azimuth" in item.properties and "view:incidence_angle" in item.properties: 
            saa.append(item.properties["view:sun_azimuth"])
            sza.append(90-item.properties["view:sun_elevation"])
            vaa.append(item.properties["view:azimuth"])
            vza.append(item.properties["view:incidence_angle"])
        else: 
            metadata = requests.get(item.assets['granule_metadata'].href).text

            root = ET.fromstring(metadata)
            child = root.find(root.tag.split('}')[0]+'}Geometric_Info')
            in_root = child.find('Tile_Angles').find('Mean_Viewing_Incidence_Angle_List')
            for angle in in_root.iter('Mean_Viewing_Incidence_Angle'):
                if angle.attrib:
                    if angle.get('bandId') == '4':
                        zenith4 = angle.find('ZENITH_ANGLE')
                        azimuth4 = angle.find('AZIMUTH_ANGLE')
                        if not (zenith4.get('unit') == 'deg' and azimuth4.get('unit') == 'deg'):
                            print('Warning: angle unit: ', zenith4.get('unit'), azimuth4.get('unit'))
                        saa.append(item.properties["view:sun_azimuth"])
                        sza.append(90-item.properties["view:sun_elevation"])
                        vza.append(float(zenith4.text))
                        vaa.append(float(azimuth4.text))

    vza = np.array(vza)
    vaa = np.array(vaa)
    sza = np.array(sza)
    saa = np.array(saa)
    return vza, vaa, sza, saa

            

def load_data_10(item):
    cube = load_data(item, bands=bands_10m, resolution=10)

    return cube

def load_data_20(item):
    cube = load_data(item, bands=bands_20m, resolution=20)

    return cube

def load_data_60(item):
    cube = load_data(item, bands=bands_60m, resolution=60)

    return cube

def load_data_none(item):
    cube = load_data(item, bands=bands_none, resolution=10)

    return cube