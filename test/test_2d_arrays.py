import pytest

import numpy as np
import xarray as xr

from xarrayMannKendall import *

n=100
time = np.arange(n)
x = np.arange(4)

data = np.zeros((len(time), len(x)))

da = xr.DataArray(data, coords=[time, x], 
                    dims=['time', 'lon'])

################################################################################
############################## Test data nan ###################################
################################################################################

def nan_in_data_MK(da):
    linear_trend = xr.DataArray(time, coords=[time], dims=['time'])
    dataarray_plus_lt = (da + linear_trend) * np.nan
    MK_class = Mann_Kendall_test(dataarray_plus_lt, 'time', coords_name = {'time':'time','lon':'x'}, MK_modified=True, method="linregress")
    MK_trends = MK_class.compute()
    return MK_trends

@pytest.mark.xrMK
def test_nan_data_MK():
    dataarray = nan_in_data_MK(da)
    assert  np.equal(dataarray.trend.values,0).all()
    assert  np.equal(dataarray.signif.values,0).all()


################################################################################
############################# No trends + noise ################################
################################################################################  

def no_trends_2d_noise(da,scale):
    noise = np.random.randn(*np.shape(da))
    dataarray_plus_lt = da * (scale * noise)
    # Additionally, test with all arguments provided (no kwargs)
    MK_class = Mann_Kendall_test(dataarray_plus_lt, 'time',
                                0.01, False, 'linregress',
                                {'time':'time','lon':'x'})
    MK_trends = MK_class.compute()
    return MK_trends

@pytest.mark.parametrize(('scale'), np.arange(1,11))

@pytest.mark.xrMK
def test_2d_no_trend_noise(scale):
    dataarray = no_trends_2d_noise(da,scale)
    np.testing.assert_almost_equal(dataarray.trend.values, 0, 2)
    assert  np.equal(dataarray.signif,0).all()
    np.testing.assert_almost_equal(dataarray.std_error.values, 0, 4)
    np.testing.assert_almost_equal(dataarray.p.values, 1, 4)

###############################################################################
############################## Trends, no noise ################################
################################################################################


def trends_2d_no_noise(da):
    linear_trend = xr.DataArray(time, coords=[time], dims=['time'])
    dataarray_plus_lt = (da + linear_trend)
    MK_class = Mann_Kendall_test(dataarray_plus_lt, 'time', coords_name = {'time':'time','lon':'x'})
    MK_trends = MK_class.compute()
    return MK_trends


@pytest.mark.xrMK
def test_2d_no_noise():
    dataarray = trends_2d_no_noise(da)
    assert  np.equal(dataarray.trend,1).all()
    assert  np.equal(dataarray.signif,1).all()
    assert  np.equal(dataarray.std_error,0).all()
    np.testing.assert_almost_equal(dataarray.p.values, 0, 4)
    #
################################################################################
############################### Trends + noise #################################
################################################################################  

def trends_2d_noise(da,scale):
    linear_trend = xr.DataArray(time, coords=[time], dims=['time'])
    noise = np.random.randn(*np.shape(data))
    dataarray_plus_lt = (da + linear_trend) + noise * scale
    MK_class = Mann_Kendall_test(dataarray_plus_lt, 'time', coords_name = {'time':'time','lon':'x'})
    MK_trends = MK_class.compute()
    return MK_trends

@pytest.mark.parametrize(('scale'), np.arange(1,10))

@pytest.mark.xrMK
def test_2d_small_noise(scale):
    dataarray = trends_2d_noise(da,scale)
    np.testing.assert_almost_equal(dataarray.trend.values, 1, 1)
    assert  np.equal(dataarray.signif,1).all()
    assert  np.not_equal(dataarray.std_error,0).all()
    np.testing.assert_almost_equal(dataarray.p.values, 0, 4)

################################################################################
########################### Modified MK + noise ################################
################################################################################

def trends_2d_noise_MK(da,scale):
    linear_trend = xr.DataArray(time, coords=[time], dims=['time'])
    noise = np.random.randn(*np.shape(data))
    dataarray_plus_lt = (da + linear_trend) + noise * scale
    MK_class = Mann_Kendall_test(dataarray_plus_lt, 'time', coords_name = {'time':'time','lon':'x'}, MK_modified=True, method="theilslopes")
    MK_trends = MK_class.compute()
    return MK_trends

@pytest.mark.parametrize(('scale'), np.arange(1,10))

@pytest.mark.xrMK
def test_2d_small_noise_MK(scale):
    dataarray = trends_2d_noise_MK(da,scale)
    np.testing.assert_almost_equal(dataarray.trend.values, 1, 1)
    assert  np.equal(dataarray.signif,1).all()
    assert  np.not_equal(dataarray.std_error,0).all()
    np.testing.assert_almost_equal(dataarray.p.values, 0, 4)