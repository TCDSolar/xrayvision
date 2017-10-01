from os import getcwd
import numpy as np
import pytest

from ..Visibility import RHESSIVisibility


class TestRHESSIVisibility(object):

    @pytest.mark.parametrize("N,M,isc,harm,erange,trange,totflux,sigamp,"
                             "chi2,xyoffset,type_string,units,"
                             "atten_state,count",
                             [(65, 65, 1, 3, [5.0, 10.0], [7.0, 19.0],
                               0.7, 0.3, 0.4, [10, 15], "photon", "test",
                               0, 72.5)])
    def test_from_map(self, N, M, isc, harm, erange, trange, totflux, sigamp,
                      chi2, xyoffset, type_string, units, atten_state, count):
        # Calculate uv and vis_in
        u, v = np.meshgrid(np.arange(M), np.arange(M))
        uv_in = np.array([u, v]).reshape(2, N*M)
        vis_in = np.zeros(N*M, dtype=complex)

        # Creating a RHESSI Visibility
        vis = RHESSIVisibility(uv_in, vis_in, isc, harm, erange, trange,
                               totflux, sigamp, chi2, xyoffset, type_string,
                               units, atten_state, count)

        assert vis.isc == isc
        assert vis.harm == harm
        assert vis.erange == erange
        assert vis.trange == trange
        assert vis.totflux == totflux
        assert vis.sigamp == sigamp
        assert vis.chi2 == chi2
        assert vis.xyoffset == xyoffset
        assert vis.type_string == type_string
        assert vis.units == units
        assert vis.atten_state == atten_state
        assert vis.count == count

    @pytest.mark.parametrize("in_str,out_str",
                             [("cm!u-2!n s!u-1!n", "cm^{-2} s^{-1}"),
                              ("m!s4", "m_{4}"),
                              ("m!n", "m"),
                              ("m!u1!u-2!n", "m^{1^{-2}}")])
    def test_unit_string_conversation(self, in_str, out_str):
        assert out_str == RHESSIVisibility.convert_units_to_tex(in_str)

    def test_fits_file_data_read_successful(self):
        print('!!!!!!!!!!!!!!!!!!!!!', getcwd(), '!!!!!!!!!!!!!!!!!')
        i = RHESSIVisibility.from_fits_file("data/hsi_20020220_110600_1time_1energy.fits")
        assert len(i) == 1

        i = RHESSIVisibility.from_fits_file("data/hsi_20020220_110600_1time_4energies.fits")
        assert len(i) == 4

        i = RHESSIVisibility.from_fits_file("data/hsi_20020220_110600_9times_1energy.fits")
        assert len(i) == 9
