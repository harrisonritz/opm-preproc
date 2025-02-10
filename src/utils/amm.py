## Python implementation of the adaptive multipole moments (tAMM) spatial filtering method.
# Copyright (C) 2018-2022 Wellcome Centre for Human Neuroimaging (GPL-2.0 license)
# Original code written by Tim Tierney: 
# code: https://github.com/spm/spm/blob/main/spm_opm_amm.m
# paper to cite: https://doi.org/10.1002/hbm.26596
# adapted for MNE-Python by Harrison Ritz (2025)

from matplotlib.pyplot import title
import numpy as np
from math import factorial, pi, sqrt
import mne
from scipy.linalg import orth



def compute_proj_amm(raw, Lout=3, Lin=5, corr=0.95):
    """
    Implements adaptive multipole moments (tAMM) spatial filtering.
    Inputs:
        raw   - a preloaded MNE Raw object.
        Lin: number of inner harmonics. (SPM default is 9, suggest 5-10)
        Lout: number of external harmonics (SPM default is 2, suggest 3-5)
        corr: correlation limit (SPM default is 1.0, suggest 0.95-0.99)
    This function adds SSP projections to the raw object to remove noise.
    """
    # Pick magnetometer channels and get data.
    picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
    data = raw.get_data(picks=picks)  # shape (n_channels, n_times)


    # Extract sensor positions and orientations from raw.info.
    pos_list, ort_list, ch_names = [], [], []
    for i in picks:
        ch = raw.info['chs'][i]
        loc = ch.get('loc', np.zeros(12))
        pos_list.append(loc[:3])
        ort_list.append(loc[3:6])
        ch_names.append(ch['ch_name'])
    positions = np.array(pos_list)    # shape (n, 3)
    orientations = np.array(ort_list)   # shape (n, 3)

    # Build sensor model dictionary M.
    M = dict()
    M['v'] = positions
    M['o'] = orientations
    M['scale'] = True  # scale positions

    # Compute outer harmonics.
    M['li'] = Lout
    M['reg'] = 1  # regular (outer)
    Sout = spm_opm_vslm(M)

    # Compute inner harmonics.
    M['li'] = Lin
    M['reg'] = 0  # irregular (inner)
    Sin = spm_opm_vslm(M)

    # Form projection operators.
    Rout = np.eye(Sout.shape[0]) - Sout @ np.linalg.pinv(Sout)
    # Solve for inner contribution.
    inneramm = Rout @ Sin @ np.linalg.pinv(Rout @ Sin) @ data
    outeramm = Sout @ np.linalg.pinv(Sout) @ data

    inner = inneramm
    outer = outeramm

    # Orthonormal bases using SVD.
    U_inner, _, _ = np.linalg.svd(inner.T, full_matrices=False)
    Oinner = U_inner
    U_outer, _, _ = np.linalg.svd(outer.T, full_matrices=False)
    Oouter = U_outer

    # NOTE: exactly recover the SPM code by using orth
    # Oinner = orth(inner.T)
    # Oouter = orth(outer.T)

    C = Oinner.T @ Oouter
    _, s, Vh = np.linalg.svd(C, full_matrices=False)
    noise = Oouter @ Vh.T
    num = np.sum(s > corr)
    if num < 1:
        num = 1
    noisevec = noise[:, :num]
    Beta = noisevec.T @ inner.T
    mod = noisevec @ Beta
    inner_tamm = inner - mod.T

    # Form SSP projections from the residual noise.
    proj_data = data - inner_tamm  # (n_channels, n_times)
    U_proj, _, _ = np.linalg.svd(proj_data, full_matrices=False)
    n_proj = num if num > 0 else 1
    proj_vecs = U_proj[:, :n_proj]

    projs = []
    for i, vec in enumerate(proj_vecs.T):
        label = f"tAMM_{i+1}"
        proj_data = dict(
            col_names=ch_names,
            row_names=None,
            data=vec[np.newaxis, :],
            ncol=len(ch_names),
            nrow=1,
        )
        projs.append(mne.Projection(active=False, data=proj_data, desc=label))


    return projs

def spm_opm_vslm(S):
    """
    Computes Cartesian real vector spherical harmonics.
    S is a dictionary with:
      'v'     - sensor positions, shape (n,3)
      'o'     - sensor orientations, shape (n,3)
      'li'    - harmonic order (integer)
      'reg'   - if 1 compute regular (outer), if 0 compute irregular (inner)
      'scale' - if True, scale positions by their maximum radius.
    Returns:
      vSlm - array of shape (n, li^2+2*li)
    """
    v = S['v'].astype(float)  # shape (n,3)
    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]
    o = S['o']
    nx = o[:, 0]
    ny = o[:, 1]
    nz = o[:, 2]

    reg = S.get('reg', 1)
    if S.get('scale', False):
        sc = np.max(np.sqrt(x**2 + y**2 + z**2))
        x = x / sc
        y = y / sc
        z = z / sc
        v = np.column_stack((x, y, z))

    li = S['li']
    n_harm = li**2 + 2 * li
    r = np.sqrt(x**2 + y**2 + z**2)
    r[r == 0] = 1e-12  # avoid div-by-zero
    atyx = np.arctan2(y, x)

    Slm = slm(v, li)  # scalar spherical harmonics, shape (n, n_harm)

    n = v.shape[0]
    Slmdx = np.zeros((n, n_harm))
    Slmdy = np.zeros((n, n_harm))
    Slmdz = np.zeros((n, n_harm))
    vSlm = np.zeros((n, n_harm))

    count = 0
    for l in range(1, li + 1):
        for m in range(-l, l + 1):
            a = (-1)**m * sqrt((2 * l + 1) / (2 * pi) * (factorial(l - abs(m)) / factorial(l + abs(m))))
            u = m * atyx
            um = abs(m) * atyx
            # L_val using associated Legendre polynomial.
            L_val = plm(z / r, l, abs(m))
            Xlm, Ylm, Zlm = dplm(v, l, abs(m))
            Xlm = np.nan_to_num(Xlm)
            Ylm = np.nan_to_num(Ylm)
            Zlm = np.nan_to_num(Zlm)
            if m < 0:
                t1 = a * np.sin(um) * Zlm
                t2 = a * L_val * abs(m) * np.cos(um) * (x / (x**2+y**2+1e-12)) + a * np.sin(um) * Ylm
                t3 = -a * L_val * abs(m) * np.cos(um) * (y / (x**2+y**2+1e-12)) + a * np.sin(um) * Xlm
            elif m == 0:
                factor0 = sqrt((2 * l + 1) / (4 * pi))
                t1 = factor0 * Zlm
                t2 = factor0 * Ylm
                t3 = factor0 * Xlm
            else:  # m > 0
                t1 = a * np.cos(u) * Zlm
                t2 = -a * L_val * m * np.sin(u) * (x / (x**2+y**2+1e-12)) + a * np.cos(u) * Ylm
                t3 = a * L_val * m * np.sin(u) * (y / (x**2+y**2+1e-12)) + a * np.cos(u) * Xlm

            if reg:
                Slmdz[:, count] = t1 * (r**l) + l * z * Slm[:, count] * (r**(l - 2))
                Slmdy[:, count] = t2 * (r**l) + l * y * Slm[:, count] * (r**(l - 2))
                Slmdx[:, count] = t3 * (r**l) + l * x * Slm[:, count] * (r**(l - 2))
            else:
                Slmdz[:, count] = t1 / (r**(l + 1)) - (l + 1) * z * Slm[:, count] / (r**(l + 3))
                Slmdy[:, count] = t2 / (r**(l + 1)) - (l + 1) * y * Slm[:, count] / (r**(l + 3))
                Slmdx[:, count] = t3 / (r**(l + 1)) - (l + 1) * x * Slm[:, count] / (r**(l + 3))
            count += 1

    for i in range(n_harm):
        vSlm[:, i] = Slmdz[:, i] * nz + Slmdy[:, i] * ny + Slmdx[:, i] * nx

    return vSlm

def slm(v, li):
    """
    Compute scalar spherical harmonics (real) for sensor positions.
    v: sensor positions, shape (n, 3)
    li: harmonic order.
    Returns:
       Slm: array of shape (n, li^2+2*li)
    """
    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    n_harm = li**2 + 2 * li
    n = v.shape[0]
    Slm = np.zeros((n, n_harm))
    count = 0
    atyx = np.arctan2(y, x)
    for l in range(1, li + 1):
        for m in range(-l, l + 1):
            if m < 0:
                L_val = plm(z / r, l, abs(m))
                Slm[:, count] = (-1)**m * sqrt((2 * l + 1) / (2 * pi) * (factorial(l - abs(m)) / factorial(l + abs(m)))) * L_val * np.sin(abs(m) * atyx)
            elif m == 0:
                L_val = plm(z / r, l, 0)
                Slm[:, count] = sqrt((2 * l + 1) / (4 * pi)) * L_val
            else:
                L_val = plm(z / r, l, m)
                Slm[:, count] = (-1)**m * sqrt((2 * l + 1) / (2 * pi) * (factorial(l - m) / factorial(l + m))) * L_val * np.cos(m * atyx)
            count += 1
    return Slm

def plm(x, l, m):
    """
    Computes the associated Legendre polynomial component.
    x: array_like input (typically z/r)
    l: degree
    m: order (nonnegative)
    Returns:
       pl: array of same shape as x.
    """
    # This is a rudimentary implementation.
    # For improved performance and accuracy consider scipy.special.lpmv.
    from scipy.special import lpmv
    # The Condon-Shortley phase (-1)**m is included in lpmv.
    pl = lpmv(m, l, x)
    return pl

def dplm(v, l, m):
    """
    Computes the partial derivatives [Xlm, Ylm, Zlm] of the associated Legendre component.
    v: sensor positions, shape (n,3)
    l: degree
    m: order (nonnegative)
    Returns:
      Xlm, Ylm, Zlm: arrays of shape (n,)
    """
    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    r[r==0] = 1e-12
    b = (-1)**m * 2**l
    n = v.shape[0]
    Xlm = np.zeros(n, dtype=float)
    Ylm = np.zeros(n, dtype=float)
    Zlm = np.zeros(n, dtype=float)
    for k in range(m, l + 1):
        factors = ((l + k - 1) / 2 - np.arange(0, l))
        val = np.prod(factors) if factors.size > 0 else 1
        if k > 0:
            vals2 = np.prod(l - np.arange(0, k))
        else:
            vals2 = 1
        c = (factorial(k) / factorial(k - m)) * (vals2 / factorial(k)) * (val / factorial(l))
        # Compute numerators elementwise.
        numx = - x * (z ** (k - m)) * (k - m) * ((x**2 + y**2) ** (m/2)) \
               + (m * x * (z ** (k - m + 2))) * ((x**2 + y**2) ** ((m - 2)/2))
        numy = - y * (z ** (k - m)) * (k - m) * ((x**2 + y**2) ** (m/2)) \
               + (m * y * (z ** (k - m + 2))) * ((x**2 + y**2) ** ((m - 2)/2))
        numz = (z ** (k - m - 1)) * (k - m) * ((x**2 + y**2) ** ((m + 2)/2)) \
               + (-m * (z ** (k - m + 1))) * ((x**2 + y**2) ** (m/2))
        numx = np.nan_to_num(numx)
        numy = np.nan_to_num(numy)
        numz = np.nan_to_num(numz)
        Xlm += b * c * numx / (r ** (2 + k))
        Ylm += b * c * numy / (r ** (2 + k))
        Zlm += b * c * numz / (r ** (2 + k))
    return Xlm, Ylm, Zlm

