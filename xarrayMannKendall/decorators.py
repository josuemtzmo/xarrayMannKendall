import functools
import numpy as np
import os
import inspect
import warnings


def dims_test(init):  
    import inspect
    arg_names = inspect.getfullargspec(init)[0]
    
    @functools.wraps(init)
    def dims_test_inner(self, *args, **kwargs):  
        for name, value in zip(arg_names[1:], args):
            setattr(self, name, value)
        
        dims = self.DataArray.dims

        if len(dims) == 1:
            raise ValueError("xarrayMannKendall requires at least a 2D dataarray (x,t)")
        elif len(dims) > 3:
            raise ValueError("Currently xarrayMannKendall only supports 2D (x,t) and 3D dataarray (x,y,t)")
        
        if 'coords_name' in dir(self) or 'coords_name' in kwargs:
            if 'coords_name' in dir(self):
                coords_name = self.coords_name
            elif 'coords_name' in kwargs:
                coords_name = kwargs['coords_name']

            if sorted(coords_name.keys()) == sorted(dims) or sorted(coords_name.values()) == sorted(dims):
                rename_dict = {(item if (key == "x" or key=="y" or key=='time') else key):
                                (key if (key == "x" or key=="y" or key=='time') else item) 
                                for key,item in coords_name.items()}
            else:
                raise ValueError("""coords_name keys {0} or items {1} do not contain 
                                    the same dimensions as the dataarray {2}""".format(coords_name.keys(),coords_name.values(),dims))

            self.DataArray = self.DataArray.rename(rename_dict)

        else:
            if len(dims) == 2 and ('x' not in dims or 'time' not in dims):
                raise ValueError('Dataarray dimensions {0} must include the dimensions (time, x)'.format(dims))
            elif len(dims) == 3 and ('x' not in dims or 'y' not in dims or 'time' not in dims):
                raise ValueError("""Dataarray is 3D {0} and must include the dimensions (time, x ,y). \n Add the 'coords_name' argument:\n coords_name = {{'time':'{1}','x':'{2}','y':'{3}'}}""".format(dims,*dims))
            
        
        self.ordered_dims = np.flipud(sorted(self.DataArray.dims))

        init(self, *args, **kwargs)

    return dims_test_inner


def check_if_file_exists(init):  
    """
    check_if_file_exists [decorator]

    This decorator handles the reload of already existing files, to avoid computing multiple times the trends.

    Parameters
    ----------
    init : [function]
    
    returns 
    """
    def get_default_args(func):
        signature = inspect.signature(func)
        return {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
    
    def reload_xarray(path):
        warnings.warn("Loading file, if you want to rewrite the file, change the argument rewrite_trends to 'False' or change the output file name.")
        import xarray as xr
        return xr.open_dataset(path)   

    @functools.wraps(init)
    def file_exists_inner(self, *args, **kwargs):

        kwds = get_default_args(init)
        kwds.update(kwargs)

        if kwds['rewrite_trends'] and kwds['path']!=None:
            if os.path.exists(kwds['path']):
                return reload_xarray(kwds['path'])
            else:
                return init(self, *args, **kwargs)
        else:
            return init(self, *args, **kwargs)
            
    return file_exists_inner
