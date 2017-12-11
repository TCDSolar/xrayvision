Introduction
============

Theory
------
Synthesis imaging relies upon describing the amplitude of some quantity on the sky (radio flux or
x-ray photon flux) in terms of complex visibilities as:

.. math:: I(x,y) = \int^{\infty}_{-\infty}\int^{\infty}_{-\infty}V(u, v)e^{-2 i \pi(ux+vy}) du dv
   :label: ifft

and the complex visibilties are given by:

.. math:: V(u,v) = \int^{\infty}_{-\infty}\int^{\infty}_{-\infty}I(x, y)e^{2 i \pi(ux+vy}) du dv
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

.. math:: B(x, y) = \sum_{i} e^{-2 i \pi(u_{i}x+v_{i}y)}w_{i}.

So the problem is to deconvolve the effects of the PSF or diry beam :math:`B` from the dirty
image :math:`I^{D}` to obtain the true image :math:`I`.

Implementation
--------------
In reality the integrals above must be turned into summations over finite coordinates so :eq:`ifft`
can be written as

.. math:: I(x_i, y_j) = \sum_{k=0}^{N} e^{2 \pi i ( x_i u_k + y_i v_k)}

where :math:`x_i`