import pygeoinf as inf
from pygeoinf.symmetric_space.plane import Sobolev as SobolevPlane
import numpy as np


def build_water_mass_forward_operator(
    mat,
    lats,
    lons,
    kmax,
    order,
    scale,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    eps=1e-8,
    parallel=False,
):
    """
    Build the forward operator mapping water mass fraction fields to point
    measurements of each observable variable.

    Args:
        mat: (nobs, nwm) end-member matrix; column j holds the observable
            values of pure water mass j.
        lats, lons: Measurement locations (y and x coordinates on the plane).
        kmax, order, scale: Sobolev plane parameters.
        lat_min, lat_max, lon_min, lon_max: Unpadded domain bounds.
        eps: NUFFT accuracy for the point evaluations.
        parallel: If True, lets finufft use all available threads.

    Returns:
        A LinearOperator from the direct sum of nwm planes to the direct sum
        of nobs EuclideanSpace(npts), i.e. data[i] = observable i at all points.
    """
    number_of_water_masses = mat.shape[1]

    assert len(lats) == len(lons), f"Got {len(lats)} lats but {len(lons)} lons"
    number_of_point_measurements = len(lats)

    plane_space = SobolevPlane(
        kmax,
        order,
        scale,
        ay=lat_min,
        by=lat_max,
        cy=scale * 10,
        ax=lon_min,
        bx=lon_max,
        cx=scale * 10,
    )
    model_space = inf.HilbertSpaceDirectSum([plane_space] * number_of_water_masses)

    # --- Build the operator that collects point measurements of each water mass at each point ---
    point_measurement_operator = build_point_measurement_operator(
        model_space,
        lats,
        lons,
        eps=eps,
        parallel=parallel,
    )

    # --- Build the operator that maps water mass concentrations to point measurements ---
    matrix_operator = build_water_mass_matrix_operator(
        mat,
        number_of_point_measurements,
    )

    # --- Compose the two operators to get the full forward operator ---
    return matrix_operator @ point_measurement_operator


def build_point_measurement_operator(model_space, lats, lons, eps=1e-8, parallel=False):
    """
    Build the operator that evaluates every water mass fraction field at the
    measurement points.

    Every field lives on the same plane and is sampled at the same points, so a
    single matrix-free (NUFFT) point evaluation operator is shared by all of
    them through a block-diagonal operator.

    Args:
        model_space: Direct sum of identical Sobolev planes, one per water mass.
        lats, lons: Measurement locations (y and x coordinates on the plane).
        eps: NUFFT accuracy for the point evaluations.
        parallel: If True, lets finufft use all available threads.

    Returns:
        A LinearOperator from model_space to the direct sum of nwm
        EuclideanSpace(npts); element j holds water mass j at all points.
    """
    plane_space = model_space.subspace(0)
    assert all(
        space == plane_space for space in model_space.subspaces
    ), "All water masses must live on the same plane"

    # Points on the plane are (x, y) = (lon, lat).
    points = list(zip(lons, lats))

    plane_to_points = plane_space.point_evaluation_operator(
        points,
        matrix_free=True,
        parallel=parallel,
        eps=eps,
    )

    return inf.BlockDiagonalLinearOperator(
        [plane_to_points] * model_space.number_of_subspaces
    )


def build_water_mass_matrix_operator(mat, number_of_point_measurements):
    """
    Build the operator that mixes water mass fractions into observables at
    every point, d_i(x_k) = sum_j mat[i, j] u_j(x_k).

    Args:
        mat: (nobs, nwm) end-member matrix.
        number_of_point_measurements: Number of measurement points, npts.

    Returns:
        A LinearOperator from the direct sum of nwm EuclideanSpace(npts) to
        the direct sum of nobs EuclideanSpace(npts).
    """
    number_of_observation_variables = mat.shape[0]
    number_of_water_masses = mat.shape[1]

    point_space = inf.EuclideanSpace(number_of_point_measurements)
    water_mass_points_space = inf.HilbertSpaceDirectSum([point_space] * number_of_water_masses)
    observation_points_space = inf.HilbertSpaceDirectSum(
        [point_space] * number_of_observation_variables
    )

    # Stack the per-water-mass arrays into an (npts, nwm) matrix and mix with
    # one small matrix product; the adjoint mixes back with mat transposed.
    def mapping(water_mass_values):
        fractions = np.column_stack(water_mass_values)
        observations = fractions @ mat.T
        return list(observations.T)

    def adjoint_mapping(observation_values):
        observations = np.column_stack(observation_values)
        fractions = observations @ mat
        return list(fractions.T)

    return inf.LinearOperator(
        water_mass_points_space,
        observation_points_space,
        mapping,
        adjoint_mapping=adjoint_mapping,
    )


def build_mass_conservation_subspace(model_space):
    """
    Build the affine subspace of water mass fraction fields that sum to one
    everywhere, sum_j u_j = 1.

    The constraint operator B(us) = sum_j u_j has B B* = nwm * I, so the
    orthogonal projector onto the tangent space is simply "subtract the mean
    field from each water mass", and no dense solve is needed. Conditioning a
    prior with identical covariance for every water mass on this subspace
    via condition_gaussian_measure(prior, geometric=True) is then exact
    Bayesian conditioning.

    Args:
        model_space: Direct sum of identical Sobolev planes, one per water mass.

    Returns:
        An AffineSubspace {us : sum_j u_j = 1}.
    """
    plane_space = model_space.subspace(0)
    assert all(
        space == plane_space for space in model_space.subspaces
    ), "All water masses must live on the same plane"
    number_of_water_masses = model_space.number_of_subspaces

    def mean_field(water_mass_fields):
        return sum(water_mass_fields) / number_of_water_masses

    # Tangent space: remove the mean field from every water mass.
    def tangent_mapping(water_mass_fields):
        mean = mean_field(water_mass_fields)
        return [field - mean for field in water_mass_fields]

    # Orthogonal complement: the mean field copied into every water mass.
    def complement_mapping(water_mass_fields):
        mean = mean_field(water_mass_fields)
        return [mean.copy() for _ in range(number_of_water_masses)]

    complement_projector = inf.LinearOperator(
        model_space,
        model_space,
        complement_mapping,
        adjoint_mapping=complement_mapping,
    )
    projector = inf.OrthogonalProjector(
        model_space,
        tangent_mapping,
        complement_projector=complement_projector,
    )

    # Equal constant fractions are a point on the subspace.
    equal_fractions = [
        plane_space.zero + 1.0 / number_of_water_masses
        for _ in range(number_of_water_masses)
    ]

    return inf.AffineSubspace(projector, translation=equal_fractions)
