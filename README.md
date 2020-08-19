# xarrayMannKendall

| Travis CI (Python 3.8) | Code Coverage |
|:----------------------:|:-------------:|
|  |  |

xarrayMannKendall is a module that allows to compute trens over 1D and 2D arrays.
For 2D arrays xarrayMannKendall uses xarray parallel capabilities to speed up the computation. 

For more information on the Mann-Kendall method please refer to:

```
Mann, H.B. 1945. Non-parametric tests against trend, Econometrica 13:163-171.
```

```
Kendall, M.G. 1975. Rank Correlation Methods, 4th edition, Charles Griffin, London.
```

```
Yue, S., & Wang, C. (2004). The Mann-Kendall Test Modified by Effective Sample Size to Detect Trend in Serially Correlated Hydrological Series. Water Resources Management, 18(3), 201–218. doi: 10.1023/b:warm.0000043140.61082.60
```
and
```
Hussain, M., & Mahmud, I. (2019). pyMannKendall: a python package for non parametric Mann Kendall family of trend tests. Journal of Open Source Software, 4(39), 1556. doi: 10.21105/joss.01556
```

An useful resource can be found [here](https://vsp.pnnl.gov/help/vsample/Design_Trend_Mann_Kendall.htm). Finally, another library that allows to compute a larger range of Mann-Kendall methods is [pyMannKendall](https://github.com/mmhs013/pyMannKendall).

This code was primarly develop for the analyisis of ocean Kinetic Energy trends 
over the satellite record. The manuscript will be available upon per-review acceptance. 
The data analysed with using this module can be found at:

[Satellite KE repository]()

## Installation:

Make sure you have the module requirements (`numpy` & `xarray`):

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