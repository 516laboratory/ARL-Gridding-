""" Imaging is based on used of the FFT to perform Fourier transforms efficiently. Since the observed visibility data
do not arrive naturally on grid points, the sampled points are resampled on the FFT grid using a convolution function to
smear out the sample points. The resulting grid points are then FFT'ed. The result can be corrected for the gridding
convolution function by division in the image plane of the transform.

This approach may be extended to include image plane effect such as the w term and the antenna/station primary beam.

This module contains functions for performing the gridding process and the inverse degridding process.
"""
from __future__ import print_function
from __future__ import absolute_import
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.curandom import rand as curand
import pycuda.gpuarray as gpuarray
from time import time
import logging

import numpy
import numpy as np
import scipy.special


from arl.fourier_transforms.fft_support import pad_mid, extract_oversampled, ifft

log = logging.getLogger(__name__)


def coordinateBounds(npixel):
    r""" Returns lowest and highest coordinates of an image/grid given:

    1. Step size is :math:`1/npixel`:

       .. math:: \frac{high-low}{npixel-1} = \frac{1}{npixel}

    2. The coordinate :math:`\lfloor npixel/2\rfloor` falls exactly on zero:

       .. math:: low + \left\lfloor\frac{npixel}{2}\right\rfloor * (high-low) = 0

    This is the coordinate system for shifted FFTs.
    """
    if npixel % 2 == 0:
        return -0.5, 0.5 * (npixel - 2) / npixel
    else:
        return -0.5 * (npixel - 1) / npixel, 0.5 * (npixel - 1) / npixel


def coordinates(npixel: int) -> object:
    """ 1D array which spans [-.5,.5[ with 0 at position npixel/2
    
    """
    return (numpy.arange(npixel) - npixel // 2) / npixel


def coordinates2(npixel: int):
    """Two dimensional grids of coordinates spanning -1 to 1 in each dimension

    1. a step size of 2/npixel and
    2. (0,0) at pixel (floor(n/2),floor(n/2))
    """
    return (numpy.mgrid[0:npixel, 0:npixel] - npixel // 2) / npixel


def coordinates2Offset(npixel: int, cx: int, cy: int):
    """Two dimensional grids of coordinates centred on an arbitrary point.
    
    This is used for A and w beams.

    1. a step size of 2/npixel and
    2. (0,0) at pixel (cx, cy,floor(n/2))
    """
    if cx is None:
        cx = npixel // 2
    if cy is None:
        cy = npixel // 2
    mg = numpy.mgrid[0:npixel, 0:npixel]
    return ((mg[0] - cy) / npixel, (mg[1] - cx) / npixel)

# @jit
def anti_aliasing_calculate(shape, oversampling=1, support=3):
    """
    Compute the prolate spheroidal anti-aliasing function //support=3 yuanlai
    
    The kernel is to be used in gridding visibility data onto a grid on for degridding from a grid.
    The gridding correction function (gcf) is used to correct the image for decorrelation due to
    gridding.
    
    Return the 2D grid correction function (gcf), and the convolving kernel (kernel

    See VLA Scientific Memoranda 129, 131, 132
    :param shape: (height, width) pair
    :param oversampling: Number of sub-samples per grid pixel
    :param support: Support of kernel (in pixels) width is 2*support+2
    """
    
    # 2D Prolate spheroidal angular function is separable
    ny, nx = shape
    nu = numpy.abs(2.0 * coordinates(nx))

    gcf1d, _ = grdsf(nu)
    gcf = numpy.outer(gcf1d, gcf1d)
    gcf[gcf > 0.0] = gcf.max() / gcf[gcf > 0.0]
    
    s1d = 2 * support + 2
    nu = numpy.arange(-support, +support, 1.0 / oversampling)
    kernel1d = grdsf(nu / support)[1]
    l1d = len(kernel1d)
    # Rearrange to get the convolution function isolated by (yf, xf). For this convolution function
    # the result is heavily redundant but it does fit well into the general framework
    kernel4d = numpy.zeros((oversampling, oversampling, s1d, s1d))
    for yf in range(oversampling):
        my = range(yf, l1d, oversampling)[::-1]
        for xf in range(oversampling):
            mx = range(xf, l1d, oversampling)[::-1]
            kernel4d[yf, xf, 2:, 2:] = numpy.outer(kernel1d[my], kernel1d[mx])
    return gcf, (kernel4d / numpy.sum(kernel4d[0,0,:,:])).astype('complex')

def grdsf(nu):
    """Calculate PSWF using an old SDE routine re-written in Python

    Find Spheroidal function with M = 6, alpha = 1 using the rational
    approximations discussed by Fred Schwab in 'Indirect Imaging'.
    This routine was checked against Fred's SPHFN routine, and agreed
    to about the 7th significant digit.
    The gridding function is (1-NU**2)*GRDSF(NU) where NU is the distance
    to the edge. The grid correction function is just 1/GRDSF(NU) where NU
    is now the distance to the edge of the image.
    """
    p = numpy.array([[8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
                     [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2]])
    q = numpy.array([[1.0000000e0, 8.212018e-1, 2.078043e-1],
                     [1.0000000e0, 9.599102e-1, 2.918724e-1]])
    
    _, np = p.shape
    _, nq = q.shape
    
    nu = numpy.abs(nu)
    
    nuend = numpy.zeros_like(nu)
    part = numpy.zeros(len(nu), dtype='int')
    part[(nu >= 0.0) & (nu < 0.75)] = 0
    part[(nu > 0.75) & (nu < 1.0)] = 1
    nuend[(nu >= 0.0) & (nu <= 0.75)] = 0.75
    nuend[(nu > 0.75) & (nu < 1.0)] = 1.0
    
    delnusq = nu ** 2 - nuend ** 2
    
    top = p[part, 0]
    for k in range(1, np):
        top += p[part, k] * numpy.power(delnusq, k)

    bot = q[part, 0]
    for k in range(1, nq):
        bot += q[part, k] * numpy.power(delnusq, k)
    # top = p[part, 0]
    # bot = q[part, 0]
    # for k in range(1, np):
    #     top += p[part, k] * numpy.power(delnusq, k)
    #     bot += q[part, k] * numpy.power(delnusq, k)

    grdsf = numpy.zeros_like(nu)
    ok = (bot > 0.0)
    grdsf[ok] = top[ok] / bot[ok]
    ok = numpy.abs(nu > 1.0)
    grdsf[ok] = 0.0
    
    # Return the gridding function and the grid correction function
    return grdsf, (1 - nu ** 2) * grdsf


def w_beam(npixel, field_of_view, w, cx=None, cy=None, remove_shift=False):
    """ W beam, the fresnel diffraction pattern arising from non-coplanar baselines
    
    :param npixel: Size of the grid in pixels
    :param field_of_view: Field of view
    :param w: Baseline distance to the projection plane
    :param cx: location of delay centre def :npixel//2
    :param cy: location of delay centre def :npixel//2
    :param remove_shift: Remove overall phase shift at the centre of the image
    :return: npixel x npixel array with the far field
    """
    if cx is None:
        cx = npixel // 2
    if cy is None:
        cy = npixel // 2
    l, m = coordinates2Offset(npixel, cx, cy)
    m *= field_of_view
    l *= field_of_view
    r2 = l ** 2 + m ** 2
    n2 = 1.0 - r2
    ph = numpy.zeros_like(n2)
    ph[r2 < 1.0] = w * (1 - numpy.sqrt(1.0 - r2[r2 < 1.0]))
    cp = numpy.zeros_like(n2, dtype='complex')
    cp[r2 < 1.0] = numpy.exp(-2j * numpy.pi * ph[r2 < 1.0])
    cp[r2 == 0] = 1.0 + 0j
    # Correct for linear phase shift in faceting
    if remove_shift:
        cp /= cp[npixel // 2, npixel // 2]
    return cp


def frac_coord(npixel, kernel_oversampling, p):
    """ Compute whole and fractional parts of coordinates, rounded to
    kernel_oversampling-th fraction of pixel size

    The fractional values are rounded to nearest 1/kernel_oversampling pixel value. At
    fractional values greater than (kernel_oversampling-0.5)/kernel_oversampling coordinates are
    rounded to next integer index.

    :param npixel: Number of pixels in total
    :param kernel_oversampling: Fractional values to round to
    :param p: Coordinate in range [-.5,.5[
    """
    assert numpy.array(p >= -0.5).all() and numpy.array(
        p < 0.5).all(), "Cellsize is too large: uv overflows grid uv= %s" % str(p)
    x = npixel // 2 + p * npixel
    flx = numpy.floor(x + 0.5 / kernel_oversampling)
    fracx = numpy.around((x - flx) * kernel_oversampling)
    return flx.astype(int), fracx.astype(int)


def convolutional_degrid(kernel_list, vshape, uvgrid, vuvwmap, vfrequencymap, vpolarisationmap=None):
    """Convolutional degridding with frequency and polarisation independent
    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param kernels: list of oversampled convolution kernel
    :param vshape: Shape of visibility
    :param uvgrid:   The uv plane to de-grid from
    :param vuvwmap: function to map uvw to grid fractions
    :param vfrequencymap: function to map frequency to image channels
    :param vpolarisationmap: function to map polarisation to image polarisation
    :return: Array of visibilities.
    """
    kernel_indices, kernels = kernel_list
    kernel_oversampling, _, gh, gw = kernels[0].shape
    assert gh % 2 == 0, "Convolution kernel must have even number of pixels"
    assert gw % 2 == 0, "Convolution kernel must have even number of pixels"
    inchan, inpol, ny, nx = uvgrid.shape
    vnpol = vshape[1]
    nvis = vshape[0]
    vis = numpy.zeros(vshape, dtype='complex')
    wt = numpy.zeros(vshape)
    
    # uvw -> fraction of grid mapping
    y, yf = frac_coord(ny, kernel_oversampling, vuvwmap[:, 1])
    y -= gh // 2
    x, xf = frac_coord(nx, kernel_oversampling, vuvwmap[:, 0])
    x -= gw // 2
    
    if len(kernels) > 1:
        coords = kernel_indices, list(vfrequencymap), x, y, xf, yf
        ckernels = numpy.conjugate(kernels)
        for pol in range(vnpol):
            vis[..., pol] = [
                numpy.sum(uvgrid[chan, pol, yy:yy+gh, xx:xx+gw] * ckernels[kind][yyf, xxf, :, :])
                for kind, chan, xx, yy, xxf, yyf in zip(*coords)
            ]
    else:
        # This is the usual case. We trim a bit of time by avoiding the kernel lookup
        coords = list(vfrequencymap), x, y, xf, yf
        ckernel0 = numpy.conjugate(kernels[0])
        for pol in range(vnpol):
            vis[..., pol] = [
                numpy.sum(uvgrid[chan, pol, yy:yy+gh, xx:xx+gw] * ckernel0[yyf, xxf, :, :])
                for chan, xx, yy, xxf, yyf in zip(*coords)
            ]
            
    return numpy.array(vis)


def convolutional_degridIndependence(kernel_list, vshape, uvgrid, vuvwmap, vfrequencymap, vpolarisationmap=None):
    kernel_indices, kernels = kernel_list
    kernel_oversampling, _, gh, gw = kernels[0].shape
    assert gh % 2 == 0, "Convolution kernel must have even number of pixels"
    assert gw % 2 == 0, "Convolution kernel must have even number of pixels"
    inchan, inpol, ny, nx = uvgrid.shape
    vnpol = vshape[1]
    nvis = vshape[0]
    vis = numpy.zeros(vshape, dtype='complex')
    wt = numpy.zeros(vshape)

    # uvw -> fraction of grid mapping
    y, yf = frac_coord(ny, kernel_oversampling, vuvwmap[:, 1])
    y -= gh // 2
    x, xf = frac_coord(nx, kernel_oversampling, vuvwmap[:, 0])
    x -= gw // 2

    if len(kernels) > 1:
        ckernels = numpy.conjugate(kernels)
        length=min(len(kernel_indices),len(vfrequencymap),len(x),
                   len(y),len(xf),len(yf))

        for pol in range(vnpol):
            for i in range(length):
                kind=kernel_indices[i]
                chan=vfrequencymap[i]
                xx=x[i]
                yy=y[i]
                xxf=xf[i]
                yyf=yf[i]
                vis[i,pol]=numpy.sum(uvgrid[chan, pol, yy:yy + gh, xx:xx + gw] * ckernels[kind][yyf, xxf, :, :])

    else:
        ckernel0 = numpy.conjugate(kernels[0])
        length=min(len(vfrequencymap),len(x),len(y),len(xf),len(yf))
        for pol in range(vnpol):
            for i in range(length):
                chan=vfrequencymap[i]
                xx=x[i]
                yy=y[i]
                xxf=xf[i]
                yyf=yf[i]
                vis[i,pol]=numpy.sum(uvgrid[chan, pol, yy:yy + gh, xx:xx + gw] * ckernel0[yyf, xxf, :, :])

    return numpy.array(vis)

def convolutional_degrid_GPU(kernel_list, vshape, uvgrid, vuvwmap, vfrequencymap, vpolarisationmap=None):
    mod=SourceModule("""
    #include<stdio.h>
    #include<stdlib.h>
    __global__ void convol_degird_kernels2(float *visReal,
                    float *visImag,
                    float *uvgridReal,
                    float *uvgridImag,
                    float *ckernel0Real,
                    float *ckernel0Imag,
                    int *vfrequencymap,
                    int *x,
                    int *y,
                    int *xf,
                    int *yf,
                    int gh,
                    int gw,
                    int nx,
                    int vnpol,
                    int length)
    {
          
          for(int pol=0;pol<vnpol;pol++)
          {
             int row=threadIdx.x+blockIdx.x*blockDim.x;
             int col=threadIdx.y+blockIdx.y*blockDim.y;
             int slience=threadIdx.z+blockIdx.z*blockDim.z;
             int i=row+col*blockDim.x*gridDim.x+slience*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
             if(i<length)
             {
                int chan=vfrequencymap[i];
                int xx=x[i];
                int yy=y[i];
                int xxf=xf[i];
                int yyf=yf[i];
                float sumReal=0.0;
                float sumImag=0.0;
          
                int t1=chan*vnpol*nx*nx+pol*nx*nx;
                int t2=yyf*gh*gh*gh+xxf*gw*gh;
                for(int j=yy;j<yy+gh;j++)
                {
                   for(int k=xx;k<xx+gw;k++)
                   {
                      int t3=t1+j*nx+k;
                      int t4=t2+(j-yy)*gh+k-xx;
                      sumReal+=(uvgridReal[t3]*ckernel0Real[t4]-uvgridImag[t3]*ckernel0Imag[t4]);
                      sumImag+=(uvgridReal[t3]*ckernel0Imag[t4]+uvgridImag[t3]*ckernel0Real[t4]);
                   }
                }
               visReal[i*vnpol+pol]=sumReal;
               visImag[i*vnpol+pol]=sumImag;
             }  
          }
    }
    """)
    kernel_indices, kernels = kernel_list
    kernel_oversampling, _, gh, gw = kernels[0].shape
    assert gh % 2 == 0, "Convolution kernel must have even number of pixels"
    assert gw % 2 == 0, "Convolution kernel must have even number of pixels"
    inchan, inpol, ny, nx = uvgrid.shape
    vnpol = vshape[1]
    nvis = vshape[0]
    vis = numpy.zeros(vshape, dtype='complex')
    wt = numpy.zeros(vshape)

    # uvw -> fraction of grid mapping
    y, yf = frac_coord(ny, kernel_oversampling, vuvwmap[:, 1])
    y -= gh // 2
    x, xf = frac_coord(nx, kernel_oversampling, vuvwmap[:, 0])
    x -= gw // 2
    uvgridReal=uvgrid.real
    uvgridImag=uvgrid.imag

    if len(kernels) > 1:
        ckernels = numpy.conjugate(kernels)
        length=min(len(kernel_indices),len(vfrequencymap),len(x),
                   len(y),len(xf),len(yf))

        for pol in range(vnpol):
            for i in range(length):
                kind=kernel_indices[i]
                chan=vfrequencymap[i]
                xx=x[i]
                yy=y[i]
                xxf=xf[i]
                yyf=yf[i]
                vis[i,pol]=numpy.sum(uvgrid[chan, pol, yy:yy + gh, xx:xx + gw] * ckernels[kind][yyf, xxf, :, :])

    else:
        ckernel0 = numpy.conjugate(kernels[0])
        ckernel0Real=ckernel0.real
        ckernel0Imag=ckernel0.imag
        length=min(len(vfrequencymap),len(x),len(y),len(xf),len(yf))

        visReal=np.zeros_like(wt,dtype=np.float32)
        visIamg=np.zeros_like(wt,dtype=np.float32)
        vis_real=visReal.reshape(-1)
        vis_iamg=visIamg.reshape(-1)
        uvgridReal=np.array(uvgridReal)
        uvgrid_real=uvgridReal.reshape(-1)
        uvgridImag=np.array(uvgridImag)
        uvgrid_imag=uvgridImag.reshape(-1)
        ckernel0Real=np.array(ckernel0Real)
        ckernel0_real=ckernel0Real.reshape(-1)
        ckernel0Imag=np.array(ckernel0Imag)
        ckernel0_imag=ckernel0Imag.reshape(-1)

        # vis_real_gpu=drv.mem_alloc_like(vis_real)
        # vis_iamg_gpu=drv.mem_alloc_like(vis_iamg)
        uvgrid_real_gpu=drv.mem_alloc_like(uvgrid_real)
        uvgrid_imag_gpu=drv.mem_alloc_like(uvgrid_imag)
        ckernel0_real_gpu=drv.mem_alloc_like(ckernel0_real)
        ckernel0_imag_gpu=drv.mem_alloc_like(ckernel0_imag)
        vfrequencymap_gpu=drv.mem_alloc_like(vfrequencymap)
        x_gpu=drv.mem_alloc_like(x)
        y_gpu=drv.mem_alloc_like(y)
        xf_gpu=drv.mem_alloc_like(xf)
        yf_gpu=drv.mem_alloc_like(yf)

        strm = drv.Stream()
        drv.memcpy_htod_async(uvgrid_real_gpu, np.array(uvgrid_real), strm)
        drv.memcpy_htod_async(uvgrid_imag_gpu, np.array(uvgrid_imag), strm)
        drv.memcpy_htod_async(ckernel0_real_gpu, np.array(ckernel0_real), strm)
        drv.memcpy_htod_async(ckernel0_imag_gpu, np.array(ckernel0_imag), strm)
        drv.memcpy_htod_async(vfrequencymap_gpu, np.array(vfrequencymap), strm)
        drv.memcpy_htod_async(x_gpu, np.array(x), strm)
        drv.memcpy_htod_async(y_gpu, np.array(y), strm)
        drv.memcpy_htod_async(xf_gpu, np.array(xf), strm)
        drv.memcpy_htod_async(yf_gpu, np.array(yf), strm)
        strm.synchronize()
        uvgrid_real = np.array(uvgrid_real)
        uvgrid_imag = np.array(uvgrid_imag)
        vis_real=np.array(vis_real)
        vis_iamg=np.array(vis_iamg)
        convol_degird_kernels2=mod.get_function("convol_degird_kernels2")

        convol_degird_kernels2(drv.Out(vis_real),
                            drv.Out(vis_iamg),
                            uvgrid_real_gpu,
                            uvgrid_imag_gpu,
                            ckernel0_real_gpu,
                            ckernel0_imag_gpu,
                            vfrequencymap_gpu,
                            x_gpu,
                            y_gpu,
                            xf_gpu,
                            yf_gpu,
                            np.int32(gh),
                            np.int32(gw),
                            np.int32(nx),
                            np.int32(vnpol),
                            np.int32(length),
                            block=(32, 32, 1),
                            grid=(1024, 96, 1)
                            )
        vis=numpy.ones((length,vnpol), dtype='complex')
        vis_real_2D=vis_real.reshape(-1,vnpol)
        vis_iamg_2D=vis_iamg.reshape(-1,vnpol)
        vis.real=vis_real_2D
        vis.imag=vis_iamg_2D
    return numpy.array(vis)


def convolutional_grid(kernel_list, uvgrid, vis, visweights, vuvwmap, vfrequencymap, vpolarisationmap=None):
    """Grid after convolving with frequency and polarisation independent gcf

    Takes into account fractional `uv` coordinate values where the GCF is oversampled

    :param kernels: List of oversampled convolution kernels
    :param uvgrid: Grid to add to [nchan, npol, npixel, npixel]
    :param vis: Visibility values
    :param visweights: Visibility weights
    :param vuvwmap: map uvw to grid fractions
    :param vfrequencymap: map frequency to image channels
    :param vpolarisationmap: map polarisation to image polarisation
    :return: uv grid[nchan, npol, ny, nx], sumwt[nchan, npol]
    """

    kernel_indices, kernels = kernel_list
    # kernels_array=np.array(kernels)
    # f=kernels_array.shape
    #f= kernels.shape
    kernel_oversampling, _, gh, gw = kernels[0].shape
    assert gh % 2 == 0, "Convolution kernel must have even number of pixels"
    assert gw % 2 == 0, "Convolution kernel must have even number of pixels"
    inchan, inpol, ny, nx = uvgrid.shape

    # Construct output grids (in uv space)
    sumwt = numpy.zeros([inchan, inpol])

    # uvw -> fraction of grid mapping
    y, yf = frac_coord(ny, kernel_oversampling, vuvwmap[:, 1])
    y -= gh // 2
    x, xf = frac_coord(nx, kernel_oversampling, vuvwmap[:, 0])
    x -= gw // 2

    # About 228k samples per second for standard kernel so about 10 million CMACs per second

    # Now we can loop over all rows
    wts = visweights[...]
    viswt = vis[...] * visweights[...]
    npol = vis.shape[-1]

    if len(kernels) > 1:
        coords = kernel_indices, list(vfrequencymap), x, y, xf, yf
        for pol in range(npol):
            for v, vwt, kind, chan, xx, yy, xxf, yyf in zip(viswt[..., pol],wts[..., pol], *coords):
                uvgrid[chan, pol, yy:yy+gh, xx:xx+gw] += kernels[kind][yyf, xxf, :, :] * v
                sumwt[chan, pol] += vwt
    else:
        kernel0 = kernels[0]
        coords = list(vfrequencymap), x, y, xf, yf
        for pol in range(npol):
            for v, vwt, chan, xx, yy, xxf, yyf in zip(viswt[..., pol], wts[..., pol], *coords):
                uvgrid[chan, pol, yy:yy+gh, xx:xx+gw] += kernel0[yyf, xxf, :, :] * v
                sumwt[chan, pol] += vwt
    return uvgrid, sumwt

def myzip(*seqs):
    minlen=min(len(s) for s in seqs)
    return [tuple(s[i] for s in seqs) for i in range(minlen)]

# def convolutional_grid_test2(kernel_list, uvgrid, vis, visweights, vuvwmap, vfrequencymap, vpolarisationmap=None):
#
#     kernel_indices, kernels = kernel_list
#     kernel_oversampling, _, gh, gw = kernels[0].shape
#     assert gh % 2 == 0, "Convolution kernel must have even number of pixels"
#     assert gw % 2 == 0, "Convolution kernel must have even number of pixels"
#     inchan, inpol, ny, nx = uvgrid.shape
#
#     # Construct output grids (in uv space)
#     sumwt = numpy.zeros([inchan, inpol])
#
#     # uvw -> fraction of grid mapping
#     y, yf = frac_coord(ny, kernel_oversampling, vuvwmap[:, 1])
#     y -= gh // 2
#     x, xf = frac_coord(nx, kernel_oversampling, vuvwmap[:, 0])
#     x -= gw // 2
#
#     # About 228k samples per second for standard kernel so about 10 million CMACs per second
#
#     # Now we can loop over all rows
#     wts = visweights[...]
#     viswt = vis[...] * visweights[...]
#     npol = vis.shape[-1]
#
#     if len(kernels) > 1:
#         coords = kernel_indices, list(vfrequencymap), x, y, xf, yf
#         for pol in range(npol):
#             seqs=myzip(viswt[..., pol], wts[..., pol], *coords)
#             for v, vwt, kind, chan, xx, yy, xxf, yyf in seqs:
#                 uvgrid[chan, pol, yy:yy + gh, xx:xx + gw] += kernels[kind][yyf, xxf, :, :] * v
#                 sumwt[chan, pol] += vwt
#
#     else:
#         kernel0 = kernels[0]
#         coords = list(vfrequencymap), x, y, xf, yf
#         for pol in range(npol):
#             seqs=myzip(viswt[..., pol],wts[..., pol], *coords)
#             for v, vwt, chan, xx, yy, xxf, yyf in seqs:
#                 uvgrid[chan, pol, yy:yy+gh, xx:xx+gw] += kernel0[yyf, xxf, :, :] * v
#                 sumwt[chan, pol] += vwt
#     return uvgrid, sumwt


# def convolutional_grid_test3(kernel_list, uvgrid, vis, visweights, vuvwmap, vfrequencymap, vpolarisationmap=None):
#
#     kernel_indices, kernels = kernel_list
#     kernel_oversampling, _, gh, gw = kernels[0].shape
#     assert gh % 2 == 0, "Convolution kernel must have even number of pixels"
#     assert gw % 2 == 0, "Convolution kernel must have even number of pixels"
#     inchan, inpol, ny, nx = uvgrid.shape
#
#     # Construct output grids (in uv space)
#     sumwt = numpy.zeros([inchan, inpol])
#
#     # uvw -> fraction of grid mapping
#     y, yf = frac_coord(ny, kernel_oversampling, vuvwmap[:, 1])
#     y -= gh // 2
#     x, xf = frac_coord(nx, kernel_oversampling, vuvwmap[:, 0])
#     x -= gw // 2
#
#     # About 228k samples per second for standard kernel so about 10 million CMACs per second
#
#     # Now we can loop over all rows
#     wts = visweights[...]
#     viswt = vis[...] * visweights[...]
#     npol = vis.shape[-1]
#     if len(kernels) > 1:
#         coords = kernel_indices, list(vfrequencymap), x, y, xf, yf
#         for pol in range(npol):
#             minlen=min(len(viswt[..., pol]),len(wts[..., pol]),len(kernel_indices),
#                        len(list(vfrequencymap)),len(x),len(y),len(xf),len(yf))
#             for i in range(minlen):
#                 v=viswt[i,:,pol]
#                 vwt=wts[i,:,pol]
#                 kind=kernel_indices[i]
#                 chan=vfrequencymap[i]
#                 xx=x[i]
#                 yy=y[i]
#                 xxf=xf[i]
#                 yyf=yf[i]
#                 uvgrid[chan, pol, yy:yy + gh, xx:xx + gw] += kernels[kind][yyf, xxf, :, :] * v
#                 sumwt[chan, pol] += vwt
#
#     else:
#         kernel0 = kernels[0]
#         coords = list(vfrequencymap), x, y, xf, yf
#         for pol in range(npol):
#             minlen=min(len(viswt[..., pol]),len(wts[..., pol]),len(vfrequencymap),
#                        len(x),len(y),len(xf),len(yf))
#             for i in range(minlen):
#                 v=viswt[i,:,pol]
#                 vwt=wts[i,:,pol]
#                 chan=vfrequencymap[i]
#                 xx=x[i]
#                 yy=y[i]
#                 xxf=xf[i]
#                 yyf=yf[i]
#                 uvgrid[chan, pol, yy:yy + gh, xx:xx + gw] += kernel0[yyf, xxf, :, :] * v
#                 sumwt[chan, pol] += vwt
#     return uvgrid, sumwt

def convolutional_grid_GPU(kernel_list, uvgrid, vis, visweights, vuvwmap, vfrequencymap, vpolarisationmap=None):
    mod=SourceModule("""
    #include<stdio.h>
    #include<stdlib.h>
    __global__ void convol_grid_kernel1(float *uvgrid_real,
               float *uvgrid_imag,
               float *sumwt,
               float *kernels_real,
               float *kernels_imag,
               float *viswt_real,
               float *viswt_imag,
               float *wts,
               int *kernel_indices,
               int *vfrequencypam,
               int *x,
               int *y,
               int *xf,
               int *yf,
               int nx,
               int gh,
               int gw,
               int npol,
               int length)
    {
        for(int pol=0;pol<npol;pol++)
        {
           int row=threadIdx.x+blockIdx.x*blockDim.x;
           int col=threadIdx.y+blockIdx.y*blockDim.y;
           int slience=threadIdx.z+blockIdx.z*blockDim.z;
           int i=row+col*blockDim.x*gridDim.x+
               slience*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
           if(i<length)
           {
              float v_real=viswt_real[i*npol+pol];
              float v_imag=viswt_imag[i*npol+pol];
              float vwt=wts[i*npol+pol];
              int kind=kernel_indices[i];
              int chan=vfrequencypam[i];
              int xx=x[i];
              int yy=y[i];
              int xxf=xf[i];
              int yyf=yf[i];
              for(int j=yy;j<yy+gh;j++)
                 for(int k=xx;k<xx+gw;k++)
                 {
                    int w=chan*npol*nx*nx+pol*nx*nx+j*nx+k;
                    int q=kind*gh*gh*gh*gh+yyf*gh*gh*gh+xxf*gh*gh+j*gh+k;
                    uvgrid_real[w] +=(kernels_real[q]*v_real-
                        kernels_imag[q]*v_imag);
                    uvgrid_imag[w] +=(kernels_real[q]*v_imag+
                        kernels_imag[q]*v_real);
                 }
              sumwt[chan*npol+pol]+=vwt;
           }
        }
    }
    
    __global__ void convol_grid_kernel2(float *uvgrid_real,
               float *uvgrid_imag,
               float *sumwt,
               float *kernel0_real,
               float *kernel0_imag,
               float *viswt_real,
               float *viswt_imag,
               float *wts,
               int *vfrequencymap,
               int *x,
               int *y,
               int *xf,
               int *yf,
               int nx,
               int gh,
               int gw,
               int npol,
               int length)
    {
       for(int pol=0;pol<npol;pol++)
       {
          int row=threadIdx.x+blockIdx.x*blockDim.x;
          int col=threadIdx.y+blockIdx.y*blockDim.y;
          int slience=threadIdx.z+blockIdx.z*blockDim.z;
          int i=row+col*blockDim.x*gridDim.x+
               slience*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
          if(i<length)
          {
             float v_real=viswt_real[i*npol+pol];
             float v_imag=viswt_imag[i*npol+pol];
             float vwt=wts[i*npol+pol];
             int chan=vfrequencymap[i];//89
             int xx=x[i];
             int yy=y[i];
             int xxf=xf[i];
             int yyf=yf[i];
             for(int j=yy;j<yy+gh;j++)
                for(int k=xx;k<xx+gw;k++)
                {
                   int w=chan*pol*nx*nx+pol*nx*nx+j*nx+k;
                   int q=yyf*gh*gh*gh+xxf*gh*gh+j*gh+k;
                   uvgrid_real[w]+=(kernel0_real[q]*v_real-
                            kernel0_imag[q]*v_imag);
                   uvgrid_imag[w]+=(kernel0_real[q]*v_imag+
                            kernel0_imag[q]*v_real);
                      
                }
             sumwt[chan*npol+pol] += vwt;
          }
       }
    }
    """)
    kernel_indices, kernels = kernel_list
    kernel_oversampling, _, gh, gw = kernels[0].shape
    assert gh % 2 == 0, "Convolution kernel must have even number of pixels"
    assert gw % 2 == 0, "Convolution kernel must have even number of pixels"
    inchan, inpol, ny, nx = uvgrid.shape

    # Construct output grids (in uv space)
    sumwt = numpy.zeros([inchan, inpol])

    # uvw -> fraction of grid mapping
    y, yf = frac_coord(ny, kernel_oversampling, vuvwmap[:, 1])
    y -= gh // 2
    x, xf = frac_coord(nx, kernel_oversampling, vuvwmap[:, 0])
    x -= gw // 2

    # About 228k samples per second for standard kernel so about 10 million CMACs per second

    # Now we can loop over all rows
    wts = visweights[...]
    viswt = vis[...] * visweights[...]
    npol = vis.shape[-1]


    uvgrid_array=np.array(uvgrid)
    uvgrid_real=uvgrid_array.real.reshape(-1)
    uvgrid_imag=uvgrid_array.imag.reshape(-1)
    viswt_array=np.array(viswt)
    viswt_real=viswt_array.real.reshape(-1)
    viswt_imag=viswt_array.imag.reshape(-1)
    wts1=np.array(wts).reshape(-1)


    if len(kernels) > 1:

        for pol in range(npol):
            minlen=min(len(viswt[..., pol]),len(wts[..., pol]),len(kernel_indices),
                       len(list(vfrequencymap)),len(x),len(y),len(xf),len(yf))
            for i in range(minlen):
                print(pol,"->",i)
                v=viswt[i,:,pol]
                vwt=wts[i,:,pol]
                kind=kernel_indices[i]
                chan=vfrequencymap[i]
                xx=x[i]
                yy=y[i]
                xxf=xf[i]
                yyf=yf[i]
                uvgrid[chan, pol, yy:yy + gh, xx:xx + gw] += kernels[kind][yyf, xxf, :, :] * v
                sumwt[chan, pol] += vwt

    else:
        kernel0 = kernels[0]

        kernel0_array=np.array(kernel0)
        kernel0_real=kernel0_array.real.reshape(-1)

        kernel0_imag=kernel0_array.imag.reshape(-1)
        convol_grid_kernel2=mod.get_function("convol_grid_kernel2")
        length = len(x)
        cublock = (32, 32, 1)
        if length // (32 * 32) > 32:
            cugrid = (32, length // (32 * 32 * 32) + 1 * (length % (32 * 32 * 32) != 0), 1)
        else:
            cugrid = (length // (32 * 32) + 1 * (length % (32 * 32) != 0), 1, 1)

        kernel0_real_gpu=drv.mem_alloc_like(kernel0_real)
        kernel0_imag_gpu=drv.mem_alloc_like(kernel0_imag)
        viswt_real_gpu=drv.mem_alloc_like(viswt_real)
        viswt_imag_gpu=drv.mem_alloc_like(viswt_imag)
        wts1_gpu=drv.mem_alloc_like(wts1)
        vfrequencymap_gpu=drv.mem_alloc_like(vfrequencymap)
        x_gpu=drv.mem_alloc_like(x)
        y_gpu=drv.mem_alloc_like(y)
        xf_gpu=drv.mem_alloc_like(xf)
        yf_gpu=drv.mem_alloc_like(yf)
        #print(kernel0_real)
        strm=drv.Stream()
        drv.memcpy_htod_async(kernel0_real_gpu,np.array(kernel0_real),strm)
        drv.memcpy_htod_async(kernel0_imag_gpu,np.array(kernel0_imag),strm)
        drv.memcpy_htod_async(viswt_real_gpu,np.array(viswt_real),strm)
        drv.memcpy_htod_async(viswt_imag_gpu,np.array(viswt_imag),strm)
        drv.memcpy_htod_async(wts1_gpu,np.array(wts1),strm)
        drv.memcpy_htod_async(vfrequencymap_gpu,np.array(vfrequencymap),strm)
        drv.memcpy_htod_async(x_gpu,np.array(x),strm)
        drv.memcpy_htod_async(y_gpu,np.array(y),strm)
        drv.memcpy_htod_async(xf_gpu,np.array(xf),strm)
        drv.memcpy_htod_async(yf_gpu,np.array(yf),strm)
        strm.synchronize()
        uvgrid_real=np.array(uvgrid_real)
        uvgrid_imag=np.array(uvgrid_imag)
        convol_grid_kernel2(drv.Out(uvgrid_real),
                            drv.Out(uvgrid_imag),
                            drv.Out(sumwt),
                            kernel0_real_gpu,
                            kernel0_imag_gpu,
                            viswt_real_gpu,
                            viswt_imag_gpu,
                            wts1_gpu,
                            vfrequencymap_gpu,
                            x_gpu,
                            y_gpu,
                            xf_gpu,
                            yf_gpu,
                            np.int32(nx),
                            np.int32(gh),
                            np.int32(gw),
                            np.int32(npol),
                            np.int32(length),
                            block=(32,32,1),
                            grid=(100,100,1)
                            )
        uvgrid_real_4D=uvgrid_real.reshape(inchan,inpol,nx,ny)
        uvgrid_imag_4D=uvgrid_imag.reshape(inchan,inpol,nx,ny)
        uvgrid.real=uvgrid_real_4D
        uvgrid.imag=uvgrid_imag_4D
    return uvgrid, sumwt


def weight_gridding(shape, visweights, vuvwmap, vfrequencymap, vpolarisationmap=None, weighting='uniform'):
    """Reweight data using one of a number of algorithms

    :param shape:
    :param visweights: Visibility weights
    :param vuvwmap: map uvw to grid fractions
    :param vfrequencymap: map frequency to image channels
    :param vpolarisationmap: map polarisation to image polarisation
    :param weighting: '' | 'uniform'
    :return: visweights, density, densitygrid
    """
    densitygrid = numpy.zeros(shape)
    density = numpy.zeros_like(visweights)
    if weighting == 'uniform':
        log.info("weight_gridding: Performing uniform weighting")
        inchan, inpol, ny, nx = shape
        
        # uvw -> fraction of grid mapping
        y, yf = frac_coord(ny, 1.0, vuvwmap[:, 1])
        x, xf = frac_coord(nx, 1.0, vuvwmap[:, 0])
        wts = visweights[...]
        coords = list(vfrequencymap), x, y
        for pol in range(inpol):
            for vwt, chan, x, y in zip(wts, *coords):
                densitygrid[chan, pol, y, x] += vwt[..., pol]
        
        # Normalise each visibility weight to sum to one in a grid cell
        newvisweights = numpy.zeros_like(visweights)
        for pol in range(inpol):
            density[..., pol] += [densitygrid[chan, pol, x, y] for chan, x, y in zip(*coords)]
        newvisweights[density > 0.0] = visweights[density > 0.0] / density[density > 0.0]
        return newvisweights, density, densitygrid
    else:
        return visweights, None, None

def visibility_recentre(uvw, dl, dm):
    """ Compensate for kernel re-centering - see `w_kernel_function`.

    :param uvw: Visibility coordinates
    :param dl: Horizontal shift to compensate for
    :param dm: Vertical shift to compensate for
    :returns: Visibility coordinates re-centrered on the peak of their w-kernel
    """

    u, v, w = numpy.hsplit(uvw, 3)
    return numpy.hstack([u - w*dl, v - w*dm, w])


def gridder(uvgrid, vis, xs, ys, kernel=numpy.ones((1, 1)), kernel_ixs=None):
    """Grids visibilities at given positions. Convolution kernels are selected per
    visibility using ``kernel_ixs``.

    :param uvgrid: Grid to update (two-dimensional :class:`complex` array)
    :param vis: Visibility values (one-dimensional :class:`complex` array)
    :param xs: Visibility position (one-dimensional :class:`int` array)
    :param ys: Visibility values (one-dimensional :class:`int` array)
    :param kernel: Convolution kernel (minimum two-dimensional :class:`complex` array).
      If the kernel has more than two dimensions, additional indices must be passed
      in ``kernel_ixs``. Default: Fixed one-pixel kernel with value 1.
    :param kernel_ixs: Map of visibilities to kernel indices (maximum two-dimensional :class:`int` array).
      Can be omitted if ``kernel`` requires no indices, and can be one-dimensional
      if only one index is needed to identify kernels
    """
    
    if kernel_ixs is None:
        kernel_ixs = numpy.zeros((len(vis), 0))
    else:
        kernel_ixs = numpy.array(kernel_ixs)
        if len(kernel_ixs.shape) == 1:
            kernel_ixs = kernel_ixs.reshape(len(kernel_ixs), 1)
    
    gh, gw = kernel.shape[-2:]
    for v, x, y, kern_ix in zip(vis, xs, ys, kernel_ixs):
        uvgrid[y:y + gh, x:x + gw] += kernel[convert_to_tuple3(kern_ix)] * v

    return uvgrid

def convert_to_tuple3(x_ary):
    """ Numba cannot do this conversion itself. Hardcode 3 for speed"""
    y_tup = ()
    y_tup += (x_ary[0],)
    y_tup += (x_ary[1],)
    y_tup += (x_ary[2],)
    return y_tup


def gridder_numba(uvgrid, vis, xs, ys, kernel=numpy.ones((1, 1)), kernel_ixs=None):
    """Grids visibilities at given positions. Convolution kernels are selected per
    visibility using ``kernel_ixs``.

    :param uvgrid: Grid to update (two-dimensional :class:`complex` array)
    :param vis: Visibility values (one-dimensional :class:`complex` array)
    :param xs: Visibility position (one-dimensional :class:`int` array)
    :param ys: Visibility values (one-dimensional :class:`int` array)
    :param kernel: Convolution kernel (minimum two-dimensional :class:`complex` array).
      If the kernel has more than two dimensions, additional indices must be passed
      in ``kernel_ixs``. Default: Fixed one-pixel kernel with value 1.
    :param kernel_ixs: Map of visibilities to kernel indices (maximum two-dimensional :class:`int` array).
      Can be omitted if ``kernel`` requires no indices, and can be one-dimensional
      if only one index is needed to identify kernels
    """
    
    # if kernel_ixs is None:
    #     kernel_ixs = numpy.zeros((len(vis), 0))
    # else:
    #     kernel_ixs = numpy.array(kernel_ixs)
    #     if len(kernel_ixs.shape) == 1:
    #         kernel_ixs = kernel_ixs.reshape(len(kernel_ixs), 1)
    
    gh, gw = kernel.shape[-2:]
    for v, x, y, kern_ix in zip(vis, xs, ys, kernel_ixs):
        uvgrid[y:y + gh, x:x + gw] += kernel[convert_to_tuple3(kern_ix)] * v
        
    return uvgrid
