"""Sampler integration, reproducibility, and synthetic posterior recovery."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm, uniform

import astropy.units as apu

from xrayvision.vis_forward_fit.bayesian import BayesianVisibilityProblem, ParameterPrior, sample_emcee
from xrayvision.vis_forward_fit.forward_fit import sources_to_vis, vis_forward_fit
from xrayvision.vis_forward_fit.sources import Source, SourceList
from xrayvision.visibility import Visibilities

emcee = pytest.importorskip("emcee")
FLUX_UNIT = apu.ph / (apu.s * apu.cm**2 * apu.keV)


def make_problem(shape="circular"):
    models = {
        "circular": Source("circular", 2 * FLUX_UNIT, -1.5 * apu.arcsec, 0.8 * apu.arcsec, 2 * apu.arcsec),
        "elliptical": Source(
            "elliptical",
            1 * FLUX_UNIT,
            4 * apu.arcsec,
            3 * apu.arcsec,
            1 * apu.arcsec,
            3 * apu.arcsec,
            30 * apu.deg,
        ),
        "loop": Source(
            "loop",
            1.5 * FLUX_UNIT,
            -3 * apu.arcsec,
            1 * apu.arcsec,
            2 * apu.arcsec,
            5 * apu.arcsec,
            20 * apu.deg,
            40 * apu.deg,
        ),
    }
    sources = SourceList(list(models.values()) if shape == "mixed" else [models[shape]])
    rng = np.random.default_rng(42)
    u, v = rng.uniform(-0.12, 0.12, (2, 30)) / apu.arcsec
    obs = Visibilities(sources_to_vis(sources, u, v), u, v, uncertainty=np.full(30, 0.03 + 0.06j) * FLUX_UNIT)
    priors = {
        name: ParameterPrior(norm(loc=value, scale=10), unit)
        for name, unit, value in zip(
            sources.parameter_names, sources.parameter_units(FLUX_UNIT), sources.to_canonical(FLUX_UNIT).params
        )
    }
    return BayesianVisibilityProblem(obs, sources, priors), obs


@pytest.fixture
def setup_sampler():
    problem, _ = make_problem()
    positions = problem.initial_params + np.random.default_rng(3).normal(0, 0.01, (12, problem.ndim))
    return problem, positions


@pytest.mark.parametrize("shape", ["circular", "elliptical", "loop", "mixed"])
def test_sampler_supports_existing_models_and_unit_aware_results(shape):
    problem, _ = make_problem(shape)
    n_walkers = 2 * problem.ndim + 2
    positions = problem.initial_params + np.random.default_rng(3).normal(0, 0.001, (n_walkers, problem.ndim))
    result = sample_emcee(problem, positions, n_steps=8, burn_in=3, seed=17)
    assert result.chain.shape == (8, n_walkers, problem.ndim)
    assert result.log_prob.shape == (8, n_walkers)
    assert result.samples.shape == (8 * n_walkers, problem.ndim)
    assert result.parameter_names == problem.parameter_names
    assert result.parameter_units == problem.parameter_units
    assert np.all(np.isfinite(result.chain))
    assert np.all(np.isfinite(result.log_prob))
    assert np.all((result.acceptance_fraction >= 0) & (result.acceptance_fraction <= 1))
    assert_allclose(result.log_prob[-1, -1], problem.log_posterior(result.chain[-1, -1]))
    assert_allclose(result.parameter_samples("source[0].x0").value, result.samples[:, 1])
    assert result.parameter_samples("source[0].amp").unit == FLUX_UNIT
    assert_allclose(
        result.credible_interval("source[0].x0", level=0.8).value,
        np.quantile(result.samples[:, 1], [0.1, 0.9]),
    )
    reconstructed = result.sources_from_sample(-1, 0)
    assert [type(source) for source in reconstructed] == [type(source) for source in problem.sources]
    assert_allclose(
        sources_to_vis(reconstructed, problem.fit_data.u, problem.fit_data.v),
        problem.predict(result.chain[-1, 0]),
    )


def test_sampler_reproducible_without_mutating_input_or_global_rng(setup_sampler):
    problem, positions = setup_sampler
    original_positions = positions.copy()
    before = np.random.get_state()
    first = sample_emcee(problem, positions, n_steps=12, burn_in=3, seed=51)
    second = sample_emcee(problem, positions, n_steps=12, burn_in=3, seed=51)
    different = sample_emcee(problem, positions, n_steps=12, burn_in=3, seed=52)
    after = np.random.get_state()
    assert_array_equal(first.chain, second.chain)
    assert_array_equal(first.log_prob, second.log_prob)
    assert not np.array_equal(first.chain, different.chain)
    assert_array_equal(positions, original_positions)
    for previous, current in zip(before, after):
        assert_array_equal(previous, current)


def test_burn_in_is_discarded_and_acceptance_counts_only_production(setup_sampler):
    problem, positions = setup_sampler
    full = sample_emcee(problem, positions, n_steps=17, seed=3)
    result = sample_emcee(problem, positions, n_steps=12, burn_in=5, seed=3)
    assert_array_equal(result.chain, full.chain[5:])
    assert_array_equal(result.log_prob, full.log_prob[5:])
    accepted = np.any(np.diff(full.chain[4:], axis=0) != 0, axis=-1)
    assert_allclose(result.acceptance_fraction, accepted.mean(axis=0))
    assert result.diagnostics["burn_in"] == 5


def test_autocorrelation_short_chain_estimates_are_flagged(setup_sampler, monkeypatch):
    problem, positions = setup_sampler

    def too_short(self):
        raise emcee.autocorr.AutocorrError(np.full(problem.ndim, 10.0), "Not enough production steps")

    monkeypatch.setattr(emcee.EnsembleSampler, "get_autocorr_time", too_short)
    result = sample_emcee(problem, positions, n_steps=3, seed=3)
    assert result.autocorrelation_status == "chain_too_short"
    assert_allclose(result.autocorrelation_time, 10)
    assert "Not enough production steps" in result.diagnostics["autocorrelation_message"]


@pytest.mark.parametrize("tau, expected", [(1.0, "estimated"), (np.nan, "unavailable"), (0.0, "unavailable")])
def test_autocorrelation_status_does_not_claim_convergence(setup_sampler, monkeypatch, tau, expected):
    problem, positions = setup_sampler
    monkeypatch.setattr(emcee.EnsembleSampler, "get_autocorr_time", lambda self: np.full(problem.ndim, tau))
    result = sample_emcee(problem, positions, n_steps=3, seed=3)
    assert result.autocorrelation_status == expected
    if expected == "unavailable":
        assert result.autocorrelation_time is None
    else:
        assert "convergence is not established" in result.diagnostics["autocorrelation_message"]


def test_single_production_step_has_no_autocorrelation_estimate(setup_sampler):
    problem, positions = setup_sampler
    result = sample_emcee(problem, positions, n_steps=1, seed=3)
    assert result.autocorrelation_status == "unavailable"
    assert result.autocorrelation_time is None


@pytest.mark.parametrize("n_steps, burn_in", [(0, 0), (-1, 0), (3, -1)])
def test_sampler_rejects_invalid_lengths(setup_sampler, n_steps, burn_in):
    problem, positions = setup_sampler
    with pytest.raises(ValueError, match="must be positive"):
        sample_emcee(problem, positions, n_steps=n_steps, burn_in=burn_in)


def test_sampler_validates_initial_ensemble(setup_sampler):
    problem, positions = setup_sampler
    with pytest.raises(ValueError, match="must have shape"):
        sample_emcee(problem, positions[:, :-1], n_steps=1)
    with pytest.raises(ValueError, match="twice as many walkers"):
        sample_emcee(problem, positions[:2], n_steps=1)
    with pytest.raises(ValueError, match="must be real"):
        sample_emcee(problem, positions.astype(complex), n_steps=1)
    with pytest.raises(TypeError, match="not a Quantity array"):
        sample_emcee(problem, positions * apu.arcsec, n_steps=1)
    invalid = positions.copy()
    invalid[0, 0] = -1
    with pytest.raises(ValueError, match="finite log posterior"):
        sample_emcee(problem, invalid, n_steps=1)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite log posterior"):
        sample_emcee(problem, invalid, n_steps=1)
    with pytest.raises(ValueError, match="condition number"):
        sample_emcee(problem, np.tile(problem.initial_params, (12, 1)), n_steps=1)


def test_result_validates_summary_requests(setup_sampler):
    problem, positions = setup_sampler
    result = sample_emcee(problem, positions, n_steps=3, seed=3)
    with pytest.raises(KeyError, match="Unknown parameter"):
        result.parameter_samples("typo")
    for level in (0, 1, -0.1, np.nan):
        with pytest.raises(ValueError, match="strictly between zero and one"):
            result.credible_interval("source[0].x0", level=level)


def test_seeded_noisy_circular_posterior_matches_local_analytic_solution():
    problem, noiseless = make_problem()
    truth = problem.initial_params
    u, v = problem.fit_data.u, problem.fit_data.v
    prediction = noiseless.visibilities.value
    derivative = np.array(
        [
            prediction / truth[0],
            2j * np.pi * u * prediction,
            2j * np.pi * v * prediction,
            -4 * np.pi**2 * truth[3] * (u**2 + v**2) * prediction,
        ]
    ).T
    sigma = np.concatenate([np.full(30, 0.03), np.full(30, 0.06)])
    jacobian = np.vstack([derivative.real, derivative.imag]) / sigma[:, None]
    covariance = np.linalg.inv(jacobian.T @ jacobian + np.eye(4) / 100)
    rng = np.random.default_rng(74)
    noise = rng.normal(size=60) * sigma
    observed = noiseless.visibilities + (noise[:30] + 1j * noise[30:]) * FLUX_UNIT
    vis = Visibilities(observed, noiseless.u, noiseless.v, uncertainty=noiseless.uncertainty)
    noisy_problem = BayesianVisibilityProblem(vis, problem.sources, problem.priors)
    expected_mean = truth + covariance @ jacobian.T @ (noise / sigma)
    positions = rng.multivariate_normal(expected_mean, covariance, size=16)
    result = sample_emcee(noisy_problem, positions, n_steps=800, burn_in=200, seed=18)
    scale = np.sqrt(np.diag(covariance))
    # At this noise level the independent linearised posterior is accurate.
    assert_allclose((result.samples.mean(axis=0) - expected_mean) / scale, 0, atol=0.3)
    assert_allclose(result.samples.std(axis=0, ddof=1) / scale, 1, rtol=0.2)
    assert np.all(np.abs(result.samples.mean(axis=0) - truth) < 4 * scale)


@pytest.mark.parametrize("sampler_seed", [18, 19])
def test_noisy_elongated_ellipse_recovers_identifiable_parameters(sampler_seed):
    reference, noiseless = make_problem("elliptical")
    truth = reference.initial_params
    u, v = reference.fit_data.u, reference.fit_data.v
    prediction = noiseless.visibilities.value
    up = u * np.cos(truth[5]) + v * np.sin(truth[5])
    vp = -u * np.sin(truth[5]) + v * np.cos(truth[5])
    # Independent analytical derivatives give the local Fisher covariance and
    # the expected response to one fixed noise realisation, including theta.
    derivative = np.array(
        [
            prediction / truth[0],
            2j * np.pi * u * prediction,
            2j * np.pi * v * prediction,
            -4 * np.pi**2 * up**2 * truth[3] * prediction,
            -4 * np.pi**2 * vp**2 * truth[4] * prediction,
            -4 * np.pi**2 * up * vp * (truth[3] ** 2 - truth[4] ** 2) * prediction,
        ]
    ).T
    sigma = np.concatenate([np.full(30, 0.01), np.full(30, 0.02)])
    jacobian = np.vstack([derivative.real, derivative.imag]) / sigma[:, None]
    covariance = np.linalg.inv(jacobian.T @ jacobian)
    # Change the sampler seed, but keep this measurement noise fixed.
    rng = np.random.default_rng(74)
    noise = rng.normal(size=60) * sigma
    observed = noiseless.visibilities + (noise[:30] + 1j * noise[30:]) * FLUX_UNIT
    vis = Visibilities(observed, noiseless.u, noiseless.v, uncertainty=np.full(30, 0.01 + 0.02j) * FLUX_UNIT)
    priors = {
        "source[0].amp": ParameterPrior(uniform(0.1, 3.9), FLUX_UNIT),
        "source[0].x0": ParameterPrior(uniform(-10, 20), apu.arcsec),
        "source[0].y0": ParameterPrior(uniform(-10, 20), apu.arcsec),
        "source[0].sigmax": ParameterPrior(uniform(0.5, 4), apu.arcsec),
        "source[0].sigmay": ParameterPrior(uniform(0.5, 4), apu.arcsec),
        "source[0].theta": ParameterPrior(uniform(-np.pi / 4, np.pi / 2), apu.rad),
    }
    problem = BayesianVisibilityProblem(vis, reference.sources, priors)
    expected_mean = truth + covariance @ jacobian.T @ (noise / sigma)
    positions = np.random.default_rng(sampler_seed).multivariate_normal(expected_mean, covariance, size=24)
    result = sample_emcee(problem, positions, n_steps=1600, burn_in=400, seed=sampler_seed)
    scale = np.sqrt(np.diag(covariance))
    assert_allclose((result.samples.mean(axis=0) - expected_mean) / scale, 0, atol=0.4)
    assert_allclose(result.samples.std(axis=0, ddof=1) / scale, 1, rtol=0.2)
    assert np.all(np.abs(result.samples.mean(axis=0) - truth) < 4 * scale)


@pytest.mark.parametrize("sampler_seed", [20, 21])
def test_circular_ellipse_recovers_size_without_identifying_orientation(sampler_seed):
    # Rotationally balanced sampling and noiseless data isolate the geometry
    # degeneracy. A noisy realisation can prefer a small apparent ellipticity.
    radii, angles = np.meshgrid([0.02, 0.05, 0.08, 0.11], np.linspace(0, 2 * np.pi, 16, endpoint=False))
    u = (radii * np.cos(angles)).ravel() / apu.arcsec
    v = (radii * np.sin(angles)).ravel() / apu.arcsec
    sources = SourceList(
        [
            Source(
                "elliptical", 2 * FLUX_UNIT, 0 * apu.arcsec, 0 * apu.arcsec, 2 * apu.arcsec, 2 * apu.arcsec, 0 * apu.rad
            )
        ]
    )
    circular = SourceList([Source("circular", 2 * FLUX_UNIT, 0 * apu.arcsec, 0 * apu.arcsec, 2 * apu.arcsec)])
    vis = Visibilities(sources_to_vis(circular, u, v), u, v, uncertainty=np.full(u.size, 0.03 + 0.03j) * FLUX_UNIT)
    priors = {
        "source[0].amp": ParameterPrior(uniform(0.1, 3.9), FLUX_UNIT),
        "source[0].x0": ParameterPrior(norm(0, 5), apu.arcsec),
        "source[0].y0": ParameterPrior(norm(0, 5), apu.arcsec),
        "source[0].sigmax": ParameterPrior(uniform(0.5, 4), apu.arcsec),
        "source[0].sigmay": ParameterPrior(uniform(0.5, 4), apu.arcsec),
        "source[0].theta": ParameterPrior(uniform(-np.pi / 4, np.pi / 2), apu.rad),
    }
    problem = BayesianVisibilityProblem(vis, sources, priors)
    params = problem.initial_params.copy()
    likelihoods = []
    for angle in np.linspace(-np.pi / 4, np.pi / 4, 25):
        params[5] = angle
        assert_allclose(problem.predict(params).value, vis.visibilities.value, atol=1e-13)
        likelihoods.append(problem.log_likelihood(params))
    assert_allclose(likelihoods, likelihoods[0], atol=1e-10)
    rng = np.random.default_rng(sampler_seed)
    positions = problem.initial_params + rng.normal(size=(24, 6)) * [0.02, 0.03, 0.03, 0.02, 0.02, 0.1]
    positions[:, 5] = rng.uniform(-np.pi / 4, np.pi / 4, 24)
    result = sample_emcee(problem, positions, n_steps=2000, burn_in=500, seed=sampler_seed)
    # Flux, location and a rotation-invariant size are identifiable even when
    # theta is not. Require both recovery and useful precision.
    size = np.sqrt((result.samples[:, 3] ** 2 + result.samples[:, 4] ** 2) / 2)
    identifiable = np.column_stack([result.samples[:, :3], size])
    lower, upper = np.quantile(identifiable, [0.025, 0.975], axis=0)
    truth = np.array([2, 0, 0, 2])
    assert np.all(lower < truth)
    assert np.all(truth < upper)
    assert np.all(upper - lower < 0.15)
    theta = result.samples[:, 5]
    assert np.ptp(np.quantile(theta, [0.025, 0.975])) > 0.8 * (np.pi / 2)
    # Fourth harmonic is appropriate to this half-pi angle interval with
    # exchangeable widths. A uniform angle distribution has zero moment.
    assert np.abs(np.mean(np.exp(4j * theta))) < 0.2


@pytest.mark.parametrize("shape, n_steps", [("loop", 1600), ("mixed", 2400)])
def test_noisy_loop_and_mixed_source_parameter_recovery(shape, n_steps):
    models = [
        Source("circular", 1.2 * FLUX_UNIT, -8 * apu.arcsec, -5 * apu.arcsec, 1.4 * apu.arcsec),
        Source(
            "elliptical",
            0.9 * FLUX_UNIT,
            7 * apu.arcsec,
            -4 * apu.arcsec,
            1.1 * apu.arcsec,
            2.3 * apu.arcsec,
            0.35 * apu.rad,
        ),
        Source(
            "loop",
            1.5 * FLUX_UNIT,
            -1 * apu.arcsec,
            6 * apu.arcsec,
            2 * apu.arcsec,
            4.5 * apu.arcsec,
            0.25 * apu.rad,
            0.9 * apu.rad,
        ),
    ]
    sources = SourceList(models[-1:] if shape == "loop" else models)
    # Resolved components and one source of each type avoid interchangeable
    # labels. Positive curvature and angles away from their chart boundaries
    # select one representation of each geometry for parameter summaries.
    # These are local recovery tests, not a test of exploring every mode.
    ranges = {
        "Circular": [[0.2, -15, -15, 0.5], [3, 15, 15, 3]],
        "Elliptical": [[0.2, -15, -15, 0.5, 0.5, -np.pi / 4], [3, 15, 15, 3, 3, np.pi / 4]],
        "Loop": [[0.2, -15, -15, 1.2, 3.5, -np.pi / 2, 0.3], [3, 15, 15, 2.8, 6, np.pi / 2, 1.2]],
    }
    rng = np.random.default_rng(91)
    u, v = rng.uniform(-0.14, 0.14, (2, 96)) / apu.arcsec
    sigma_real, sigma_imag = 0.005, 0.008
    noise = rng.normal(0, sigma_real, 96) + 1j * rng.normal(0, sigma_imag, 96)
    vis = Visibilities(
        sources_to_vis(sources, u, v) + noise * FLUX_UNIT,
        u,
        v,
        uncertainty=np.full(96, sigma_real + 1j * sigma_imag) * FLUX_UNIT,
    )
    truth = np.asarray(sources.to_canonical(FLUX_UNIT).params)
    guess = truth.copy()
    offset = 0
    for source in sources:
        guess[offset : offset + 3] += [-0.06, 0.2, -0.15]
        guess[offset + 3 : offset + source.n_params] *= 1.04
        offset += source.n_params
    initial = SourceList.from_params(sources, guess)
    for source in initial:
        source.amp *= FLUX_UNIT
        source.bounds = ranges[type(source).__name__]
    # The fixed physical ranges above define explicit uniform priors. Match
    # the optimiser bounds for this experiment without deriving scientific
    # priors from the fitted parameters or their uncertainty estimates.
    lower, upper = np.asarray(initial.bounds)
    priors = {
        name: ParameterPrior(uniform(loc=lo, scale=hi - lo), unit)
        for name, unit, lo, hi in zip(initial.parameter_names, initial.parameter_units(FLUX_UNIT), lower, upper)
    }
    _, fitted, fit = vis_forward_fit(
        vis,
        initial,
        [33, 33] * apu.pix,
        [1, 1] * apu.arcsec / apu.pix,
        map=False,
        method="L-BFGS-B",
        return_result=True,
        uncertainty="gauss_newton",
    )
    assert fit.success
    assert fit.parameter_uncertainty.status == "ok"
    covariance = fit.parameter_uncertainty.covariance
    assert covariance is not None
    centre = np.asarray(fitted.to_canonical(FLUX_UNIT).params)
    problem = BayesianVisibilityProblem(vis, fitted, priors)
    # Start around the perturbed-guess SciPy fit, not at the injected vector.
    # At this noise level the local Gauss-Newton covariance approximates the
    # posterior, but the sampler must recover its location and spread itself.
    sampler_rng = np.random.default_rng(73)
    positions = sampler_rng.multivariate_normal(centre, 4 * covariance, size=2 * problem.ndim + 6)
    result = sample_emcee(problem, positions, n_steps=n_steps, burn_in=500, seed=52)
    scale = np.sqrt(np.diag(covariance))
    assert_allclose((result.samples.mean(axis=0) - centre) / scale, 0, atol=0.4)
    assert_allclose(result.samples.std(axis=0, ddof=1) / scale, 1, rtol=0.25)
    # Noise can displace a recovered parameter from its injected value. These
    # are individual checks, not a claim of joint credible-interval coverage.
    assert np.all(np.abs(result.samples.mean(axis=0) - truth) < 4 * scale)

    # Check recovery of the combined visibility field at baselines not used in
    # fitting, so success is more than agreeing at the observed sample points.
    test_u, test_v = rng.uniform(-0.14, 0.14, (2, 32)) / apu.arcsec
    expected = sources_to_vis(sources, test_u, test_v).to_value(FLUX_UNIT)
    indices = sampler_rng.choice(len(result.samples), size=200, replace=False)
    predictions = np.array(
        [
            sources_to_vis(
                result.sources_from_sample(int(i // result.chain.shape[1]), int(i % result.chain.shape[1])),
                test_u,
                test_v,
            ).to_value(FLUX_UNIT)
            for i in indices
        ]
    )
    residual = predictions.mean(axis=0) - expected
    assert np.sqrt(np.mean(residual.real**2)) < 2 * sigma_real
    assert np.sqrt(np.mean(residual.imag**2)) < 2 * sigma_imag
