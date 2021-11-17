import mock
import pytest

@pytest.mark.xrMK
def test_init():
  from xarrayMannKendall import xarrayMannKendall
  with mock.patch.object(xarrayMannKendall, "__name__", "__main__"):
        xarrayMannKendall.init("__main__")