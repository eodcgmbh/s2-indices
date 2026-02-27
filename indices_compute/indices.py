import numpy as np
import xarray as xr
import dask.array as da


def evi(cube_10, cube_20=None):
    # (2.5*(B08 - B04)/((B08 + 6*B04-7.5 * B02) + 1))
    b02 = cube_10.blue.astype(int)*0.0001-0.1
    b04 = cube_10.red.astype(int)*0.0001-0.1
    b08 = cube_10.nir.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = (cube_10.red).astype("bool")
        return (2.5*(b08 - b04)/((b08 + 6*b04-7.5*b02) + 1)).where(NaN, np.nan).astype("float16")

    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)
        return np.where(NaN, (2.5*(b08 - b04)/((b08 + 6*b04-7.5*b02) + 1)), np.nan).astype("float16")


def nbr(cube_10, cube_20):
    # ((B08 - B12)/(B08 + B12))
    b08 = cube_10.nir.astype(int)*0.0001-0.1
    b12 = cube_20.swir22.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = (cube_10.red).astype("bool")
        return ((b08 - b12)/(b08 + b12)).where(NaN, np.nan).astype("float16")
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)
        return np.where(NaN, ((b08 - b12)/(b08 + b12)), np.nan).astype("float16")


def ndmi(cube_10, cube_20):
    # ((B08 - B11)/(B08 + B11))
    b08 = cube_10.nir.astype(int)*0.0001-0.1
    b11 = cube_20.swir16.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = (cube_10.red).astype("bool")
        return ((b08 - b11)/(b08 + b11)).where(NaN, np.nan).astype("float16")
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)
        return np.where(NaN, ((b08 - b11)/(b08 + b11)), np.nan).astype("float16")


def nmdi(cube_10, cube_20):
    # (B08 – (B11 – B12))/(B08 + (B11 – B12))
    b08 = cube_10.nir.astype(int)*0.0001-0.1
    b11 = cube_20.swir16.astype(int)*0.0001-0.1
    b12 = cube_20.swir22.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = (cube_10.red).astype("bool")
        return ((b08 - (b11 - b12))/(b08 + (b11 - b12))).where(NaN, np.nan).astype("float16")
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)
        return np.where(NaN, ((b08 - (b11 - b12))/(b08 + (b11 - b12))), np.nan).astype("float16")


def ndwi(cube_10, cube_20=None):
    # ((B03 - B08)/(B08 + B03))
    b03 = cube_10.green.astype(int)*0.0001-0.1
    b08 = cube_10.nir.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = (cube_10.red).astype("bool")
        return ((b03 - b08)/(b08 + b03)).where(NaN, np.nan).astype("float16")
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)
        return np.where(NaN, ((b03 - b08)/(b08 + b03)), np.nan).astype("float16")


def ndii(cube_10, cube_20):
    # ((B08 - B11)/(B08 + B11))
    b08 = cube_10.nir.astype(int)*0.0001-0.1
    b11 = cube_20.swir16.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = (cube_10.red).astype("bool")
        return ((b08 - b11)/(b08 + b11)).where(NaN, np.nan).astype("float16")
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)
        return np.where(NaN, ((b08 - b11)/(b08 + b11)), np.nan).astype("float16")


def exg(cube_10, cube_20=None):
    # ((2 * B03) - (B04 + B02))
    b02 = cube_10.blue.astype(int)*0.0001-0.1
    b03 = cube_10.green.astype(int)*0.0001-0.1
    b04 = cube_10.red.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = (cube_10.red).astype("bool")
        return ((2 * b03) - (b04 + b02)).where(NaN, np.nan).astype("float16")
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)
        return np.where(NaN, ((2 * b03) - (b04 + b02)), np.nan).astype("float16")


def tcari_osavi(cube_10, cube_20):
    # (3*((B05 – B04) – 0.2 * (B05 – B03) * (B05/4))/(1.16 * B08 – (B04/B08) + B04 + 0.16)

    b03 = cube_10.green.astype(int)*0.0001-0.1
    b04 = cube_10.red.astype(int)*0.0001-0.1
    b08 = cube_10.nir.astype(int)*0.0001-0.1
    b05 = cube_20.rededge1.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = (cube_10.red).astype("bool")
        return (3*((b05-b04) - 0.2*(b05-b03)*(b05/4))/(1.16*b08 - (b04/b08) + b04 + 0.16)).where(NaN, np.nan).astype("float16")
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)
        return np.where(NaN, (3*((b05-b04) - 0.2*(b05-b03)*(b05/4))/(1.16*b08 - (b04/b08) + b04 + 0.16)), np.nan).astype("float16")


def ndvi(cube_10, cube_20=None):
    # ((B08 - B04)/(B08 + B04))
    b04 = cube_10.red.astype(int)*0.0001-0.1
    b08 = cube_10.nir.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = (cube_10.red).astype("bool")
        return ((b08 - b04)/(b08 + b04)).where(NaN, np.nan).astype("float16")
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)
        return np.where(NaN, ((b08 - b04)/(b08 + b04)), np.nan).astype("float16")


def albedo(cube_10, cube_20):
    # B02 * 0.1836 + B03 * 0.1759 + B04 * 0.1456 + B05 * 0.1347 + B06 * 0.1233 + B07 * 0.1134 + B08 * 0.1001 + B11 * 0.0231 + B12 * 0.0003
    b02 = cube_10.blue.astype(int)*0.0001-0.1
    b03 = cube_10.green.astype(int)*0.0001-0.1
    b04 = cube_10.red.astype(int)*0.0001-0.1
    b08 = cube_10.nir.astype(int)*0.0001-0.1
    b05 = cube_20.rededge1.astype(int)*0.0001-0.1
    b06 = cube_20.rededge2.astype(int)*0.0001-0.1
    b07 = cube_20.rededge3.astype(int)*0.0001-0.1
    b11 = cube_20.swir16.astype(int)*0.0001-0.1
    b12 = cube_20.swir22.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = (cube_10.red).astype("bool")
        return (b02*0.1836 + b03*0.1759 + b04*0.1456 + b05*0.1347 + b06*0.1233 + b07*0.1134 + b08*0.1001 + b11*0.0231 + b12*0.0003).where(NaN, np.nan).astype("float16")
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)
        return np.where(NaN, (b02*0.1836 + b03*0.1759 + b04*0.1456 + b05*0.1347 + b06*0.1233 + b07*0.1134 + b08*0.1001 + b11*0.0231 + b12*0.0003), np.nan).astype("float16")


def get_bands(cube_10, cube_20):
    b02 = cube_10.blue.astype(int)*0.0001-0.1
    b03 = cube_10.green.astype(int)*0.0001-0.1
    b04 = cube_10.red.astype(int)*0.0001-0.1
    b08 = cube_10.nir.astype(int)*0.0001-0.1
    b05 = cube_20.rededge1.astype(int)*0.0001-0.1
    b06 = cube_20.rededge2.astype(int)*0.0001-0.1
    b07 = cube_20.rededge3.astype(int)*0.0001-0.1
    b8a = cube_20.nir08.astype(int)*0.0001-0.1
    b11 = cube_20.swir16.astype(int)*0.0001-0.1
    b12 = cube_20.swir22.astype(int)*0.0001-0.1

    return b02, b03, b04, b08, b05, b06, b07, b8a, b11, b12


### LAI
def check_input(b3, b4, b8, lai_thresholds):
    b3_b4 = (b3 <= lai_thresholds[0]).where(b4 <= lai_thresholds[1], False)
    b3_b4_b8 = b3_b4.where(b8 <= lai_thresholds[2], False)
    return b3_b4_b8.where(b8 >= lai_thresholds[3], False)


def normalize_data(b3, b4, b8, iza, sza, siaa, xmin, xmax):
    b3 = 2 * (b3 - xmin[0]) / (xmax[0] - xmin[0]) - 1
    b4 = 2 * (b4 - xmin[1]) / (xmax[1] - xmin[1]) - 1
    b8 = 2 * (b8 - xmin[2]) / (xmax[2] - xmin[2]) - 1

    iza = 2 * (iza - xmin[3]) / (xmax[3] - xmin[3]) - 1
    sza = 2 * (sza - xmin[4]) / (xmax[4] - xmin[4]) - 1
    siaa = 2 * (siaa - xmin[5]) / (xmax[5] - xmin[5]) - 1

    return b3, b4, b8, iza, sza, siaa


def lai(cube_10, iza, sza, saa, iaa, sat):
    import math

    b03 = (cube_10.green.astype(int)-1000)/10000
    b04 = (cube_10.red.astype(int)-1000)/10000
    b08 = (cube_10.nir.astype(int)-1000)/10000
    iza = math.cos(math.radians(float(iza)))
    sza = math.cos(math.radians(float(sza)))
    siaa = math.cos(math.radians(float(saa)) - math.radians(float(iaa)))

    if sat == "s2a":
        LAI_thresholds = [0.258293766, 0.31567611, 0.770810832, 0]
        xmin = [0.0, 0.0, 0.008560811, 0.9796248, 0.342108564, -0.999999999]
        xmax = [0.248293766, 0.30567611, 0.760810832, 1, 0.927484749, 1]
        weights_first = [
            [0.631232654, 1.521456954, -1.11728837, 0.070565659, -0.210100753, -0.050903578],
            [-0.99419775, 1.351428009, 0.682282725, 0.805184201, -0.137840226, -0.602358856],
            [0.224382116, -0.066683707, -1.239754378, -0.015576853, 0.112222978, 0.03636393],
            [-0.966688976, 0.099948366, -0.313320371, 1.106360868, 0.739904923, -1.480412968],
            [-0.243396458, -0.418138872, -0.71077365, 0.811567336, 0.552429193, 1.172822939]
        ]
        bias_first = [1.468073176, 0.914303376, 1.11779465, -1.887740246, -1.44203981]
        weight_second = [-0.344628741, 0.034191041, -0.737283566, 0.001258092, 0.003956794]
        bias_second = 0.00472943

    if sat == "s2b":
        LAI_thresholds = [0.253380256, 0.305127708, 0.763770452, 0.0112776794599999]
        xmin = [0.00000000, 0.00000000, 0.02127768, 0.97962480, 0.34210856, -1.00000000]
        xmax = [0.2433803, 0.2951277, 0.7537705, 1.0000000, 0.9274847, 1.0000000]
        weights_first = [
            [1.318039736, -1.045080989, -2.026538492, 1.089475156, 1.505746683, -0.892149948],
            [-1.78455835, 0.729123273, -0.665565795, 1.448202217, -1.869293758, 1.756114215],
            [0.865092856, 1.12560905, -1.349094359, -0.008278575, -0.111830556, -0.022280943],
            [0.064097434, 0.23117271, 1.343292596, 0.02851235, -0.16910015, -0.030874576],
            [1.711389981, 2.288335798, -3.427748905, 0.377522608, 2.395064478, -1.975727207]
        ]
        bias_first = [-2.949814486, 1.222115532, 1.354398296, -1.451921349, 2.703037837]
        weight_second = [0.001687629, -0.007123092, -0.44286921, 0.950414728, 0.009208394]
        bias_second = 0.366665557

    if sat == "s2c":
        LAI_thresholds = [0.253380256, 0.305127708, 0.763770452, 0.011277679]
        xmin = [0.0, 0.0, 0.021277679, 0.9796248, 0.342108564, -0.999999999]
        xmax = [0.243380256, 0.295127708, 0.753770452, 1, 0.927484749, 1]
        weights_first = [
            [1.318039736, -1.045080989, -2.026538492, 1.089475156, 1.505746683, -0.892149948],
            [-1.78455835, 0.729123273, -0.665565795, 1.448202217, -1.869293758, 1.756114215],
            [0.865092856, 1.12560905, -1.349094359, -0.008278575, -0.111830556, -0.022280943],
            [0.064097434, 0.23117271, 1.343292596, 0.02851235, -0.16910015, -0.030874576],
            [1.711389981, 2.288335798, -3.427748905, 0.377522608, 2.395064478, -1.975727207]
        ]
        bias_first = [-2.949814486, 1.222115532, 1.354398296, -1.451921349, 2.703037837]
        weight_second = [0.001687629, -0.007123092, -0.44286921, 0.950414728, 0.009208394]
        bias_second = 0.366665557


    ymin = 0.000233774
    ymax = 13.83459255

    min_val = 0
    max_val = 8
    tolerance = -0.2

    masks = check_input(b03, b04, b08, LAI_thresholds)

    b03, b04, b08, iza, sza, siaa = normalize_data(b03, b04, b08, iza, sza, siaa, xmin, xmax)

    neuron0 = (
        b03 * weights_first[0][0] 
        + b04 * weights_first[0][1] 
        + b08 * weights_first[0][2] 
        + iza * weights_first[0][3]
        + sza * weights_first[0][4]
        + siaa * weights_first[0][5]
        + bias_first[0])

    neuron1 = (
        b03 * weights_first[1][0] 
        + b04 * weights_first[1][1] 
        + b08 * weights_first[1][2] 
        + iza * weights_first[1][3]
        + sza * weights_first[1][4]
        + siaa * weights_first[1][5]
        + bias_first[1])

    neuron2 = (
        b03 * weights_first[2][0] 
        + b04 * weights_first[2][1] 
        + b08 * weights_first[2][2] 
        + iza * weights_first[2][3]
        + sza * weights_first[2][4]
        + siaa * weights_first[2][5]
        + bias_first[2])

    neuron3 = (
        b03 * weights_first[3][0] 
        + b04 * weights_first[3][1] 
        + b08 * weights_first[3][2] 
        + iza * weights_first[3][3]
        + sza * weights_first[3][4]
        + siaa * weights_first[3][5]
        + bias_first[3])

    neuron4 = (
        b03 * weights_first[4][0] 
        + b04 * weights_first[4][1] 
        + b08 * weights_first[4][2] 
        + iza * weights_first[4][3]
        + sza * weights_first[4][4]
        + siaa * weights_first[4][5]
        + bias_first[4])

    n0_tan = da.tanh(neuron0.astype("float"))
    n1_tan = da.tanh(neuron1.astype("float"))
    n2_tan = da.tanh(neuron2.astype("float"))
    n3_tan = da.tanh(neuron3.astype("float"))
    n4_tan = da.tanh(neuron4.astype("float"))

    second_layer_output = (
        n0_tan * weight_second[0]
        + n1_tan * weight_second[1]
        + n2_tan * weight_second[2]
        + n3_tan * weight_second[3]
        + n4_tan * weight_second[4]
        + bias_second)

    denorm_val = (second_layer_output + 1) / 2 * (ymax - ymin) + ymin

    zeros = denorm_val.where(denorm_val > min_val, 0)
    zeros = zeros.where(denorm_val < max_val, 0)
    min_mask = zeros.where(denorm_val > min_val + tolerance, np.nan)
    max_mask = min_mask.where(denorm_val < max_val - tolerance, np.nan)
    lai_value = max_mask.where(masks, np.nan)

    return lai_value