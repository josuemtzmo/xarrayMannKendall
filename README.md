


# xarrayMannKendall

| Conda | Travis CI (Python 3.8) | Code Coverage | Zenodo |
|:-----:|:----------------------:|:-------------:|:------:|
| [![conda-forge](https://img.shields.io/conda/vn/conda-forge/xarrayMannKendall.svg)](https://anaconda.org/conda-forge/xarrayMannKendall) | [![Test](https://github.com/josuemtzmo/xarrayMannKendall/actions/workflows/test.yml/badge.svg)](https://github.com/josuemtzmo/xarrayMannKendall/actions/workflows/test.yml) | [![codecov](https://codecov.io/gh/josuemtzmo/xarrayMannKendall/branch/master/graph/badge.svg?token=KaUrfwvzf8)](https://codecov.io/gh/josuemtzmo/xarrayMannKendall) | [![DOI](https://zenodo.org/badge/288618695.svg)](https://zenodo.org/badge/latestdoi/288618695) |

`xarrayMannKendall` is a module to compute linear trends over 2D and 3D arrays.
For 2D arrays `xarrayMannKendall` uses [xarray](http://xarray.pydata.org/) parallel capabilities to speed up the computation. 

For more information on the Mann-Kendall method please refer to:

> Mann, H. B. (1945). Non-parametric tests against trend, *Econometrica*, **13**, 163-171.

> Kendall, M. G. (1975). Rank Correlation Methods, 4th edition, Charles Griffin, London.

> Yue, S. and Wang, C. (2004). The Mann-Kendall test modified by effective sample size to detect trend in serially correlated hydrological series. *Water Resources Management*, **18(3)**, 201–218. doi:[10.1023/b:warm.0000043140.61082.60](https://doi.org/10.1023/b:warm.0000043140.61082.60)

and

> Hussain, M. and Mahmud, I. (2019). pyMannKendall: a python package for non parametric Mann Kendall family of trend tests. *Journal of Open Source Software*, **4(39)**, 1556. doi:[10.21105/joss.01556](https://doi.org/10.21105/joss.01556)


A useful resource can be found [here](https://vsp.pnnl.gov/help/vsample/Design_Trend_Mann_Kendall.htm). Finally, another library that allows to compute a larger range of Mann-Kendall methods is [pyMannKendall](https://github.com/mmhs013/pyMannKendall).

This package was primarily developed for the analyisis of ocean Kinetic Energy trends 
over the satellite record period that can be found at doi:[10.1038/s41558-021-01006-9](https://doi.org/10.1038/s41558-021-01006-9).

The data analysed with using this module can be found at [`EKE_SST_trends`](https://github.com/josuemtzmo/EKE_SST_trends) repository.

## Installation:

You can install the latest tagged release of this package via `conda-forge` by:

```
conda install -c conda-forge xarrayMannKendall
```

Alternatively, you can clone the repository and install. To do so, make sure you
have the module requirements (`numpy` & `xarray`):

```
pip install -r requirements.txt 
```

```
conda install --file ./requirements.txt
```

Now you can install the module

```
pip install -e .
```

for local installation use 

```
pip install --ignore-installed --user .
```

## Cite this code:

This repository can be cited as:

Josué Martínez Moreno, & Navid C. Constantinou. (2021, January 23). josuemtzmo/xarrayMannKendall: Mann Kendall significance test implemented in xarray. (Version v.1.0.0). Zenodo. http://doi.org/10.5281/zenodo.4458777
