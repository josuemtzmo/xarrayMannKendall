import xarray as xr
import numpy as np
import scipy.stats as sstats
import dask.array as dsa

from xarrayMannKendall.decorators import dims_test, check_if_file_exists

class Mann_Kendall_test(object):
    """
    Compute linear trends and significance using Mann Kendall test.

    Parameters
    ----------
    DataArray : xarray.DataArray
        Dataset to analyse.
    dim : str
        Coordiante name in which the linear trend will apply ('time').
    alpha: float
        Significance level (default = 0.01)
    MK_modified: Boolean
        Modified Mann-Kendall using Yue and Wang (2004) method.
        DOI: https://doi.org/10.1023/B:WARM.0000043140.61082.60
    method: str
        Method for linear regresion: linregress (default) and theilslopes
    coords_name: dict
        Coordinates name dict renames coordinates to 'lon','lat'. 
        Example:   
            coords_name={'xu_ocean':'lon','yu_ocean':'lat','t':time}
            
    Example:
    
        # Time series length
        n=100
        time = np.arange(n)
        # Grid
        x = np.arange(4)
        y = np.arange(4)
        
        # Create dataarray
        data = np.zeros((len(time), len(x), len(y)))
        da = xr.DataArray(data, coords=[time, x , y], 
                            dims=['time', 'lon', 'lat'])
        # Create noise
        noise = np.random.randn(*np.shape(data))
        
        # Create dataarray with positive linear trend
        linear_trend = xr.DataArray(time, coords=[time], dims=['time'])
        
        # Add noise to trend
        da_with_linear_trend = (da + linear_trend) + noise
        
        # Compute trends using Mann-Kendall test
        MK_class = Mann_Kendall_test(da_with_linear_trend, 'time')
        MK_trends = MK_class.compute()        
        
    """
    
    @dims_test
    def __init__(self,DataArray,dim='time',alpha=0.01,MK_modified=False,method='linregress',coords_name=None):
        self.dim = dim
        self.alpha = alpha
        self.method = method 
        # Use if y is correlated.
        self.MK_modified = MK_modified
        
    def MK_score(self):
        r"""
        Compute Mann Kendall score. This function constructs the sign matrix. 
        Which corresponds to the number of positive differences minus the number 
        of negative differences over time.
        above the diagonal.
        $ S = \sum^{n-1}_{k-1} \sum^{n}_{j-k+1} sgn(x_j -x_k)$
        """
        # Matrix of signs
        sign_matrix=np.sign(np.subtract.outer(self.y[1:], self.y[:-1]).T) 
        # Extract values above diagonal.note that J > K always.
        score = np.sum(np.triu(sign_matrix,-1)) 
        return score
        
    def VAR_score(self):
        """
        Compute variance of S.
        """
        unique_y,count_y=np.unique(np.round(self.y,5),return_counts=True)
        g = len(unique_y) 
        count_y=count_y[count_y>1]
        
        if self.n == g:
            var_s = (self.n*(self.n-1)*(2*self.n+5))/18
        else:
            var_s = (self.n*(self.n-1)*(2*self.n+5) - np.sum(count_y*(count_y-1)*(2*count_y+5)))/18 
        return var_s
    
    def Z_test_score(self):
        """
        Compute the MK test statistic.
        """
        if self.score > 0:
            z = (self.score - 1)/np.sqrt(self.var_s)
        elif self.score  == 0 :
            z = 0
        elif self.score  < 0: 
            z = (self.score + 1)/np.sqrt(self.var_s)
        return z
    
    def P_value(self):
        """
        Compute significance (p) and hypotesis (h).
            h = true (false) if trends are significant (insignificant).
        """
        p = 2*(1-sstats.norm.cdf(abs(self.z_test)))
        h = abs(self.z_test) > sstats.norm.ppf(1-self.alpha/2)
        return p,h
    
    def _pre_computation(self,y):
        # Remove nan values of record.
        x = np.arange(0,len(y))
        self.y = y[np.isfinite(y)]#_remove_nan(y)

        self.x = x[np.isfinite(y)]#_remove_nan(x,y)
        self.n=len(self.y)
        
    def _check_length(self):
        # Make sure that x and y have the same length and they are not empty.
        if not self.x.size > 0 or not self.y.size > 0 or self.n ==0:
            self.x = np.array([0,0])
            self.y = np.array([0,0])
            self.n = len(self.y)

    def _auto_correlation(self,y_detrend,nlags):
        y = y_detrend - y_detrend.mean()
        d = self.n * np.ones(2 * self.n -1)
        auto_cov = (np.correlate(y, y, 'full') / d)[self.n - 1:]
        return auto_cov[:nlags+1]/auto_cov[0]
    
    def _calc_slope_MK(self,y,effective_n=False):
        '''
        Wrapper that returns the slope and significance using Mann-Kendall
        https://vsp.pnnl.gov/help/Vsample/Design_Trend_Mann_Kendall.htm
        '''
        self._pre_computation(y)
        self._check_length()
        
        self.score = self.MK_score()
        
        self.var_s = self.VAR_score()

        if self.method == 'linregress':
            slope,intercept = sstats.linregress(self.x, self.y)[0:2]
        elif self.method == 'theilslopes':
            slope,intercept = sstats.mstats.theilslopes(self.y, self.x, self.alpha)[0:2]
        else:
            raise ValueError('Define a method')

        y_detrend = self.y - (self.x * slope + intercept)

        if self.MK_modified or effective_n:
            ## Compute modified MK using Yue and Wang (2004) method
            acorrf = self._auto_correlation(y_detrend, nlags=self.n-1)
            idx = np.arange(1,self.n)
            sni = np.sum((1 - idx/self.n) * acorrf[idx])
            n_star = 1 + 2 * sni
            self.var_s = self.var_s * n_star
            ESS = self.n/n_star
            
        self.z_test = self.Z_test_score()
        p,h = self.P_value()

        std_error = self._calc_standard_error(y_detrend)
        
        if effective_n:
            return slope,h,p,std_error,ESS
        else:
            return slope,h,p,std_error

    def xarray_MK_trend(self):
        """
        Computes linear trend over 'dim' of xr.dataarray.
        Slope and intercept of the least square fit are added to a 
        array which contains the slope, significance mask and p-test.
        """
        da=self.DataArray.copy().transpose(*self.ordered_dims)
        axis_num = da.get_axis_num(self.dim)

        data = dsa.apply_along_axis(self._calc_slope_MK, axis_num, da.data, 
                                    dtype=np.float64,shape=(4,))
        
        return data
    
    def _calc_standard_error(self,y):
        """
        Wrapper that returns the standard_error using the effective sample size 
        computed from a modified Mann-Kendall test
        """
        std = np.std(y)
        return std/self.n
    
    @check_if_file_exists
    def compute(self,save=False,path=None,rewrite_trends=True):
        """
        Wrapper to compute trends and returns a xr.Dataset contining 
        the slope, significance mask and p-test.
        """
        trend_method=self.xarray_MK_trend()
        
        MK_output = trend_method.compute()
        
        if len(self.ordered_dims) == 2:
            ds = xr.Dataset({'trend': (['x'], MK_output[:,0]),
                         'signif': (['x'], MK_output[:,1]),
                         'p': (['x'], MK_output[:,2]),
                         'std_error': (['x'], MK_output[:,3])
                        },
                        coords={'x': (['x'], da2py(self.DataArray.x, False))})
        else:
            ds = xr.Dataset({'trend': (['y', 'x'], MK_output[:,:,0]),
                         'signif': (['y', 'x'], MK_output[:,:,1]),
                         'p': (['y', 'x'], MK_output[:,:,2]),
                         'std_error': (['y', 'x'], MK_output[:,:,3])
                        },
                        coords={'x': (['x'], da2py(self.DataArray.x,False)),
                                'y': (['y'], da2py(self.DataArray.y,False))})
        if path != None:
            ds.to_netcdf(path)
        elif save and path == None:
            ds.to_netcdf('./tmp.nc')
        return ds

def da2py(v, include_dims=True):
    if isinstance(v, xr.DataArray):
        if include_dims:
            return (v.dims, v.values)
        else:
            return v.values
    return v

def init(__name__):
    if __name__ == "__main__":
        """
        Example of implementation
        """
        #from xarrayMannKendall import *
        import numpy as np
        import xarray as xr
        n=100
        time = np.arange(n)
        x = np.arange(4)
        y = np.arange(4)
        data = np.zeros((len(time), len(x), len(y)))

        da = xr.DataArray(data, coords=[time, x , y], 
                            dims=['time', 'x', 'y'])

        noise = np.random.randn(*np.shape(data))
        linear_trend = xr.DataArray(time, coords=[time], dims=['time'])
        da_with_linear_trend = (da + linear_trend) + noise
        MK_class = Mann_Kendall_test(da_with_linear_trend, 'time')
        MK_class.compute(path='test.nc')

init(__name__)