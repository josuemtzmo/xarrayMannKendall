import pytest

import numpy as np
import xarray as xr

from xarrayMannKendall import *

n=100
time = np.arange(n)
x = np.arange(4)
y = np.arange(4)
z = np.arange(4)

################################################################################
############################### Test 1D array ##################################
################################################################################  

data = np.zeros((len(time)))

da_1d = xr.DataArray(data, coords=[time], 
                    dims=['time'])


def dimension_1D(da_1d):
    dataarray_plus_lt = da_1d 
    MK_class = Mann_Kendall_test(dataarray_plus_lt, 'time')
    MK_trends = MK_class.compute()
    return MK_trends

@pytest.mark.xrMK
def test_smaller_dimension_size():
    with pytest.raises(ValueError, match=r".* least a 2D dataarray .*"):
        dimension_1D(da_1d)


################################################################################
############################### Test 4D array ##################################
################################################################################  

data = np.zeros((len(time), len(x), len(y), len(z)))

da_4d = xr.DataArray(data, coords=[time, x , y, z], 
                    dims=['time', 'lon', 'lat', 'depth']) 

def dimension_4D(da_4d):
    dataarray_plus_lt = da_4d 
    MK_class = Mann_Kendall_test(dataarray_plus_lt, 'time')
    MK_trends = MK_class.compute()
    return MK_trends

@pytest.mark.xrMK
def test_larger_dimension_size():
    with pytest.raises(ValueError, match=r".* only supports 2D .*"):
        dimension_4D(da_4d)


################################################################################
################################ Test names ####################################
################################################################################  

data = np.zeros((len(time), len(x)))

da_names_2D = xr.DataArray(data, coords=[time, x ], 
                    dims=['time', 'lon'])

def dimension_name_2D(da_names_2D):
    dataarray_plus_lt = da_names_2D 
    MK_class = Mann_Kendall_test(dataarray_plus_lt, 'time')
    MK_trends = MK_class.compute()
    return MK_trends

@pytest.mark.xrMK
def test_dimension_name_2D():
    with pytest.raises(ValueError, match=r".* must include the dimensions .*"):
        dimension_name_2D(da_names_2D)


################################################################################
################################ Test names ####################################
################################################################################  

data = np.zeros((len(time), len(x), len(y)))

da_names = xr.DataArray(data, coords=[time, x , y], 
                    dims=['time', 'lon', 'lat'])

def dimension_name_3D(da_names):
    dataarray_plus_lt = da_names 
    MK_class = Mann_Kendall_test(dataarray_plus_lt, 'time')
    MK_trends = MK_class.compute()
    return MK_trends

@pytest.mark.xrMK
def test_dimension_name_3D():
    with pytest.raises(ValueError, match=r".*Dataarray is 3D .*"):
        dimension_name_3D(da_names)

################################################################################
################################ Test names ####################################
################################################################################  

def dimension_name(da_names):
    dataarray_plus_lt = da_names 
    MK_class = Mann_Kendall_test(dataarray_plus_lt, 'time', coords_name={'time':'time','lon':'x','y':'y'})
    MK_trends = MK_class.compute()
    return MK_trends

@pytest.mark.xrMK
def test_dimension_name():
    with pytest.raises(ValueError, match=r".* same dimensions as the .*"):
        dimension_name(da_names)


################################################################################
############################### Test reload ####################################
################################################################################  
n=100
time = np.arange(n)
x = np.arange(4)

data = np.zeros((len(time), len(x)))

da_reload = xr.DataArray(data, coords=[time,x], 
                    dims=['time','lon'])

def dimension_reload(da_reload):
    MK_class = Mann_Kendall_test(da_reload, 'time', 
                        coords_name = {'time':'time','lon':'x'})
    MK_trends = MK_class.compute(path='./test.nc')
    MK_trends_reload = MK_class.compute(path='./test.nc')
    return MK_trends,MK_trends_reload

@pytest.mark.xrMK
def test_smaller_dimension_size():
    MK_1, MK2=dimension_reload(da_reload)
    assert  np.equal(MK_1,MK2)

################################################################################
############################## Test no method ##################################
################################################################################

def no_method(data):
    MK_class = Mann_Kendall_test(data, 'time', 
                        coords_name = {'time':'time','lon':'x'},
                        method="")
    MK_trends = MK_class.compute()
    return MK_trends

@pytest.mark.xrMK
def test_smaller_dimension_size():
    with pytest.raises(ValueError, match=r"Define a method"):
        no_method(da_reload)