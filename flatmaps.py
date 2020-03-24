from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.io import fits
from astropy.wcs import WCS


def _wrap_ra(ra):
    if ra >= 360:
        return _wrap_ra(ra-360)
    elif ra < 0:
        return _wrap_ra(ra+360)
    else:
        return ra


class FlatMapInfo(object):
    def __init__(self, wcs, nx=None, ny=None,
                 lx=None, ly=None):
        """
        Creates a flat map
        :param wcs: WCS object containing information about reference
        point and resolution
        :param nx,ny: Number of pixels in the x/y axes. If None, dx/dy
        must be provided
        :param lx,ly: Extent of the map in the x/y axes. If None, nx/ny
        must be provided
        """
        self.wcs = wcs.copy()

        if nx is None and lx is None:
            raise ValueError("Must provide either nx or lx")

        if ny is None and ly is None:
            raise ValueError("Must provide either ny or ly")

        if nx is None:
            self.lx = lx
            self.nx = int(self.lx/np.abs(self.wcs.wcs.cdelt[0]))+1
        else:
            self.nx = nx
            self.lx = np.fabs(nx*self.wcs.wcs.cdelt[0])
        self.dx = self.lx/self.nx

        if ny is None:
            self.ly = ly
            self.ny = int(self.ly/np.abs(self.wcs.wcs.cdelt[1]))+1
        else:
            self.ny = ny
            self.ly = np.fabs(ny*self.wcs.wcs.cdelt[1])
        self.dy = self.ly/self.ny

        self.npix = self.nx*self.ny

    def is_map_compatible(self, mp):
        return self.npix == len(mp)

    def get_dims(self):
        """
        Returns map size
        """
        return [self.ny, self.nx]

    def get_size(self):
        """
        Returns map size
        """
        return self.npix

    def pos2pix(self, ra, dec):
        """
        Returns pixel indices for arrays of x and y coordinates.
        Will return -1 if (x,y) lies outside the map
        """
        ra = np.asarray(ra)
        scalar_input = False
        if ra.ndim == 0:
            ra = ra[None]
            scalar_input = True

        dec = np.asarray(dec)
        if dec.ndim == 0:
            dec = dec[None]

        if len(ra) != len(dec):
            raise ValueError("ra and dec must have the same size!")

        ix, iy = self.wcs.wcs_world2pix(np.array([ra, dec]).T, 0).T
        ix = ix.astype(int)
        iy = iy.astype(int)
        ix_out = np.where(np.logical_or(ix < 0,
                                        ix >= self.nx))[0]
        iy_out = np.where(np.logical_or(iy < 0,
                                        iy >= self.ny))[0]

        ipix = ix+self.nx*iy
        ipix[ix_out] = -1
        ipix[iy_out] = -1

        if scalar_input:
            return np.squeeze(ipix)
        return ipix

    def pos2pix2d(self, ra, dec):
        """
        Returns pixel indices for arrays of x and y coordinates.
        """
        ra = np.asarray(ra)
        scalar_input = False
        if ra.ndim == 0:
            ra = ra[None]
            scalar_input = True

        dec = np.asarray(dec)
        if dec.ndim == 0:
            dec = dec[None]

        if len(ra) != len(dec):
            raise ValueError("ra and dec must have the same size!")

        ix, iy = self.wcs.wcs_world2pix(np.array([ra, dec]).T, 0).T
        ix_out = np.where(np.logical_or(ix < -self.nx,
                                        ix >= 2*self.nx))[0]
        iy_out = np.where(np.logical_or(iy < -self.ny,
                                        iy >= 2*self.ny))[0]

        is_in = np.ones(len(ix), dtype=bool)
        is_in[ix_out] = False
        is_in[iy_out] = False
        is_in[np.isnan(ix)] = False
        is_in[np.isnan(iy)] = False

        if scalar_input:
            return np.squeeze(ix), np.squeeze(iy), np.squeeze(is_in)
        return ix, iy, is_in

    def pix2pos(self, ipix):
        """
        Returns x,y coordinates of pixel centres for a set of
        pixel indices.
        """
        ipix = np.asarray(ipix)
        scalar_input = False
        if ipix.ndim == 0:
            ipix = ipix[None]
            scalar_input = True

        i_out = np.where(np.logical_or(ipix < 0,
                                       ipix >= self.npix))[0]
        if len(i_out) > 0:
            print(ipix[i_out])
            raise ValueError("Pixels outside of range")

        ix = ipix % self.nx
        ioff = np.array(ipix-ix)
        iy = ioff.astype(int)/(int(self.nx))
        ix = ix.astype(np.float_)
        iy = iy.astype(np.float_)

        ra, dec = self.wcs.wcs_pix2world(np.array([ix, iy]).T, 0).T

        if scalar_input:
            return np.squeeze(ra), np.squeeze(dec)
        return ra, dec

    def get_empty_map(self):
        """
        Returns a map full of zeros
        """
        return np.zeros(self.npix, dtype=float)

    def view_map(self, map_in, ax=None, xlabel='RA', ylabel='Dec',
                 title=None, addColorbar=True,
                 cmap=cm.viridis, colorMax=None, colorMin=None,
                 fnameOut=None):
        """
        Plots a 2D map (passed as a flattened array)
        """
        if len(map_in) != self.npix:
            raise ValueError("Input map doesn't have the correct size")

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection=self.wcs)
        if title is not None:
            ax.set_title(title, fontsize=15)
        image = ax.imshow(map_in.reshape([self.ny, self.nx]),
                          vmin=colorMin, vmax=colorMax,
                          origin='lower', interpolation='nearest',
                          cmap=cmap)
        image.cmap.set_under('#777777')
        if addColorbar:
            plt.colorbar(image)
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        if fnameOut is not None:
            plt.savefig(fnameOut, bbox_inches='tight')

    def write_flat_map(self, filename, maps, descript=None):
        """
        Saves a set of maps in FITS format wit WCS.
        """

        if maps.ndim < 1:
            raise ValueError("Must supply at least one map")
        if maps.ndim == 1:
            maps = np.array([maps])
        if len(maps[0]) != self.npix:
            raise ValueError("Map doesn't conform to this pixelization")
        if descript is not None:
            if len(maps) == 1:
                descript = [descript]
            if len(maps) != len(descript):
                raise ValueError("Need one description per map")

        header = self.wcs.to_header()
        hdus = []
        for im, m in enumerate(maps):
            head = header.copy()
            if descript is not None:
                head['DESCR'] = (descript[im], 'Description')
            if im == 0:
                hdu = fits.PrimaryHDU(data=m.reshape([self.ny, self.nx]),
                                      header=head)
            else:
                hdu = fits.ImageHDU(data=m.reshape([self.ny, self.nx]),
                                    header=head)
            hdus.append(hdu)
        hdulist = fits.HDUList(hdus)
        hdulist.writeto(filename, overwrite=True)

    def u_grade(self, mp, x_fac, y_fac=None):
        """
        Up-grades the resolution of a map and returns the associated
        FlatSkyInfo object.
        mp : input map
        :param x_fac: the new map will be sub-divided into x_fac*nx
            pixels in the x direction
        :param y_fac: the new map will be sub-divided into y_fac*ny
            pixels in the y direction if y_fac=None, then y_fac=x_fac
        """
        if y_fac is None:
            y_fac = x_fac
        if len(mp) != self.npix:
            raise ValueError("Input map has a wrong size")

        w = WCS(naxis=2)
        w.wcs.cdelt = [self.wcs.wcs.cdelt[0]/int(x_fac),
                       self.wcs.wcs.cdelt[1]/int(y_fac)]
        w.wcs.crval = self.wcs.wcs.crval
        w.wcs.ctype = self.wcs.wcs.ctype
        w.wcs.crpix = [self.wcs.wcs.crpix[0]*int(x_fac),
                       self.wcs.wcs.crpix[1]*int(y_fac)]

        fm_ug = FlatMapInfo(w, nx=self.nx*int(x_fac),
                            ny=self.ny*int(y_fac))
        mp_ug = np.repeat(np.repeat(mp.reshape([self.ny, self.nx]),
                                    int(y_fac), axis=0),
                          int(x_fac), axis=1).flatten()
        return fm_ug, mp_ug

    def d_grade(self, mp, x_fac, y_fac=None):
        """
        Down-grades the resolution of a map and returns the
        associated FlatSkyInfo object.
        mp : input map
        :param x_fac: the new map will be sub-divided into
            floor(nx/x_fac) pixels in the x direction
        :param y_fac: the new map will be sub-divided into
            floor(ny/y_fac) pixels in the y direction
            if y_fac=None, then y_fac=x_fac.
        Note that if nx/ny is not a multiple of x_fac/y_fac,
        the remainder pixels will be lost.
        """
        if y_fac is None:
            y_fac = x_fac
        if len(mp) != self.npix:
            raise ValueError("Input map has a wrong size")
        print(x_fac, y_fac)
        print(int(x_fac), int(y_fac))

        w = WCS(naxis=2)
        w.wcs.cdelt = [self.wcs.wcs.cdelt[0]*int(x_fac),
                       self.wcs.wcs.cdelt[1]*int(y_fac)]
        w.wcs.crval = self.wcs.wcs.crval
        w.wcs.ctype = self.wcs.wcs.ctype
        w.wcs.crpix = [self.wcs.wcs.crpix[0]/int(x_fac),
                       self.wcs.wcs.crpix[1]/int(y_fac)]

        nx_new = self.nx/int(x_fac)
        ix_max = nx_new*int(x_fac)
        ny_new = self.ny/int(y_fac)
        iy_max = ny_new*int(y_fac)

        mp2d = mp.reshape([self.ny, self.nx])[:iy_max, :][:, :ix_max]
        fm_dg = FlatMapInfo(w, nx=nx_new, ny=ny_new)
        mp_dg = np.mean(np.mean(np.reshape(mp2d.flatten(),
                                           [ny_new,
                                            int(y_fac),
                                            nx_new,
                                            int(x_fac)]),
                                axis=-1),
                        axis=-2).flatten()

        return fm_dg, mp_dg


def read_flat_map(filename, i_map=0, hdu=None):
    """
    Reads a flat-sky map and the details of its pixelization scheme.
    The latter are returned as a FlatMapInfo object.
    :param i_map: map to read. If -1, all maps will be read.
    """
    if hdu is None:
        hdul = fits.open(filename)
        w = WCS(hdul[0].header)

        if i_map == -1:
            maps = np.array([h.data for h in hdul])
            nm, ny, nx = maps.shape
            maps = maps.reshape([nm, ny*nx])
        else:
            maps = hdul[i_map].data
            ny, nx = maps.shape
            maps = maps.flatten()
    else:
        if isinstance(hdu, list):
            w = WCS(hdu[0].header)
            maps = np.array([h.data for h in hdu])
            nm, ny, nx = maps.shape
            maps = maps.reshape([nm, ny*nx])
        else:
            w = WCS(hdu.header)
            maps = hdu.data
            ny, nx = maps.shape
            maps = maps.flatten()

    fmi = FlatMapInfo(w, nx=nx, ny=ny)

    return fmi, maps


def compare_infos(fsk1, fsk2):
    """Checks whether two FlatMapInfo objects are compatible"""
    if ((fsk1.nx != fsk2.nx) or
        (fsk1.ny != fsk2.ny) or
        (fsk1.lx != fsk2.lx) or
        (fsk1.ly != fsk2.ly)):
        raise ValueError("Map infos are incompatible")
