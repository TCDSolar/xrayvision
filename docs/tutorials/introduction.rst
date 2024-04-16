Introduction
============

Below is a figure inspired by `Dale Gary`_, which summarises the problem XRAYVISION tries to solve.
The top row shows a source map, the point spread function (PSF) or dirty beam and the convolution of
two, the dirty map (left to right). The bottom row shows the corresponding visibilities,
notice the convolution is replaced by multiplication in frequency space. The problem is given the
measured visibilities in f can the original map a be recovered?

.. plot::

    import numpy as np

    from astropy import units as u
    from astropy.convolution import convolve, Gaussian2DKernel
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    from xrayvision import transform
    from xrayvision import clean


    def make_data(noisy=False):
        g1 = Gaussian2DKernel(2, x_size=65, y_size=65).array
        g2 = Gaussian2DKernel(6, x_size=65, y_size=65).array

        temp = np.zeros((65, 65))
        temp[50,50] = 400.0

        data = convolve(temp, g1)

        temp = np.zeros((65, 65))
        temp[20,20] =2600.0

        data = data + convolve(temp, g2);

        if noisy:
            data = data + np.random.normal(loc=0.0, scale=0.005, size=(65, 65));

        return data

    data = make_data()

    full_uv = transform.generate_uv(65)

    uu = transform.generate_uv(33)
    vv = transform.generate_uv(33)

    uu, vv = np.meshgrid(uu, vv)

    # uv coordinates require unit
    uv = np.array([uu, vv]).reshape(2, 33**2)/u.arcsec

    full_vis = transform.dft_map(data, u=uv[0,:], v=uv[1,:])

    res = transform.idft_map(full_vis, u=uv[0,:], v=uv[1,:], shape=(33, 33))
    # assert np.allclose(data, res)

    # Generate log spaced radial u, v sampeling
    half_log_space = np.logspace(np.log10(np.abs(uv[uv != 0]).value.min()), np.log10(np.abs(uv.value.max())), 10)

    theta = np.linspace(0, 2*np.pi, 31)
    theta = theta[np.newaxis,:]
    theta = np.repeat(theta, 10, axis=0)

    r = half_log_space
    r = r[:, np.newaxis]
    r = np.repeat(r, 31, axis=1)

    x = r * np.sin(theta)
    y = r * np.cos(theta)


    sub_uv = np.vstack([x.flatten(), y.flatten()])
    sub_uv = np.hstack([sub_uv, np.zeros((2,1))])/u.arcsec

    sub_vis = transform.dft_map(data, u=sub_uv[0,:], v=sub_uv[1,:])

    psf1 = transform.idft_map(np.full(sub_vis.size, 1), u=sub_uv[0,:], v=sub_uv[1,:],
                              shape=(65, 65))

    sub_res = transform.idft_map(sub_vis, u=sub_uv[0,:], v=sub_uv[1,:], shape=(65, 65))

    xp = np.round(x * 33 + 33/2 - 0.5 + 16).astype(int)
    yp = np.round(y * 33 + 33/2 - 0.5 + 16).astype(int)

    sv = np.zeros((65, 65))
    sv[:,:] = 0
    sv[xp.flatten(), yp.flatten()] = 1

    v = np.pad(np.abs(full_vis.reshape(33, 33))**2, ((16, 16),(16,16)), mode='constant', constant_values=0.11)

    s_v = v*sv
    s_v[s_v == 0] = 0.11

    def im_plot(axis, data, *, text, label, **imshow_kwargs):
        axis.text(0.5, 0.85, text, fontsize=15, color='white', transform=axis.transAxes,
        horizontalalignment='center', verticalalignment='bottom', fontweight='bold')
        axis.text(0.05, 0.9, label, fontsize=14, color='white', transform=axis.transAxes)
        axis.imshow(data, origin='lower', **imshow_kwargs)
        axis.axis('off')

    f, (r1, r2) = plt.subplots(2, 3, figsize=(12,8))

    im_plot(r1[0], data, text=r'$I(l, m)$', label='a )')
    im_plot(r1[1], psf1, text=r'$B(l, m)$', label='b )')
    im_plot(r1[2], sub_res, text=r'$I(l, m) *B(l, m)$', label='c )')
    im_plot(r2[0], v, text=r'$V(u, v)$', norm=LogNorm(0.1), label='d )')
    im_plot(r2[1], np.ones((65,65)), text=r'$S(u, v)$', label='e )', extent=(-1, 1, -1, 1))
    r2[1].plot(x.flatten(), y.flatten(), 'w.', ms=2.5)
    im_plot(r2[2], s_v, text=r'$S(u,v)V(u, v)$', label='f )', extent=(-1, 1, -1, 1), norm=LogNorm(0.1))

    f.subplots_adjust(hspace=0.05, wspace=0.025)
    plt.show()

Theory
------
Synthesis imaging relies upon describing the amplitude of some quantity on the sky (radio flux or
x-ray photon flux) in terms of complex visibilities as:

.. math:: I(l,m) = \int^{\infty}_{-\infty}\int^{\infty}_{-\infty}V(u, v)e^{-2 i \pi(ul+vm}) du dv
   :label: ifft

and the complex visibilties are given by:

.. math:: V(u,v) = \int^{\infty}_{-\infty}\int^{\infty}_{-\infty}I(l, m)e^{2 i \pi(ux+vy}) dl dm
   :label: fft

In the case where the :math:`u, v` plane is fully sampled the amplitude can be retrieved by simple
inversion. Any real instrument only nosily samples the :math:`u, v` plane. Ignoring noise this
sampling function can repented as a series of delta functions and written as

.. math:: S(u,v) = \sum_{i} w_{i} \delta (u-u_{i}) \delta ( v - v_{i})

substituting this into :eq:`ifft` we obtain the dirty image

.. math:: I^{D} = \mathscr{F}^{-1} SV

applying the convolution theorem

.. math:: I^{D} = B * I

where :math:`B = \mathscr{F}^{-1} S` is the point spread function (PSF) also known as the dirty
beam given by

.. math:: B(l, m) = \sum_{i} e^{-2 i \pi(u_{i}l+v_{i}m)}w_{i}.

So the problem is to deconvolve the effects of the PSF or diry beam :math:`B` from the dirty
image :math:`I^{D}` to obtain the true image :math:`I`.

Implementation
--------------
In reality the integrals above must be turned into summations over finite coordinates so :eq:`ifft`
can be written as

.. math:: I(l_i, m_j) = \sum_{k=0}^{N} e^{2 \pi i ( l_i u_k + m_i v_k)}

where :math:`x_i`

.. _Dale Gary: https://web.njit.edu/~gary/728/Lecture6.html
