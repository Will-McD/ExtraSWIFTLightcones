#!/bin/env python
"""
Tests swiftlet.beam_plotting's functions to draw beams on a Mollweide map of the sky:
    - sky_coordinates gives the longitude and latitude the close repeat maps use
    - the edge of a beam is at its angular radius from the axis
    - the area drawn holds exactly the directions inside the beam, for beams crossing the edge of the map and around
      a pole
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from matplotlib.path import Path  # noqa: E402

from swiftlet.beam_plotting import beam_sky_outline, plot_beam_on_sky, sky_coordinates  # noqa: E402

# (beam vector, angular radius [deg]): on an axis, across the edge of the map (longitude pi), around each pole, small
BEAMS = [
    pytest.param((1.0, 0.0, 0.0), 20.0, id="x_axis"),
    pytest.param((-1.0, 0.02, 0.2), 25.0, id="across_the_edge"),
    pytest.param((0.2, 0.1, 1.0), 30.0, id="around_north_pole"),
    pytest.param((0.0, -0.3, -1.0), 40.0, id="around_south_pole"),
    pytest.param((1.0, 2.0, 3.0), 2.0, id="small"),
]


def directions(longitude, latitude):
    """Unit vectors of directions on the sky, from their longitude and latitude [rad]."""
    return np.column_stack(
        [np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)]
    )


def test_sky_coordinates():
    """
    sky_coordinates gives arctan2(y, x) and arcsin(z / r), whatever the distance, as the close repeat maps use.
    """
    rng = np.random.default_rng(1)
    positions = rng.normal(size=(1000, 3)) * rng.uniform(1.0, 3000.0, size=(1000, 1))
    longitude, latitude = sky_coordinates(positions)
    r = np.linalg.norm(positions, axis=1)
    np.testing.assert_allclose(longitude, np.arctan2(positions[:, 1], positions[:, 0]))
    np.testing.assert_allclose(latitude, np.arcsin(positions[:, 2] / r))
    np.testing.assert_allclose(directions(longitude, latitude), positions / r[:, None], atol=1e-12)


@pytest.mark.parametrize("beam_vector, ang_radius_deg", BEAMS)
def test_edge_is_at_the_angular_radius(beam_vector, ang_radius_deg):
    """
    Every point of the edge is the beam's angular radius from its axis, and the edge is closed.
    """
    longitude, latitude = beam_sky_outline(beam_vector, ang_radius_deg)
    axis = np.asarray(beam_vector) / np.linalg.norm(beam_vector)
    angle = np.degrees(np.arccos(np.clip(directions(longitude, latitude) @ axis, -1.0, 1.0)))
    np.testing.assert_allclose(angle, ang_radius_deg, atol=1e-9)
    np.testing.assert_allclose(
        directions(longitude[[0]], latitude[[0]]), directions(longitude[[-1]], latitude[[-1]]), atol=1e-12
    )


@pytest.mark.parametrize("beam_vector, ang_radius_deg", BEAMS)
def test_area_drawn_is_the_beam(beam_vector, ang_radius_deg):
    """
    A direction is inside the area drawn (in one of its copies shifted by 2 pi) exactly if it is inside the beam,
    away from the edge.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="mollweide")
    drawn = plot_beam_on_sky(ax, beam_vector, ang_radius_deg)
    areas = [Path(artist.get_xy()) for artist in drawn if isinstance(artist, Polygon)]
    assert len(areas) == 3 and len(drawn) == 6
    plt.close(fig)

    rng = np.random.default_rng(2)
    longitude = rng.uniform(-np.pi, np.pi, 20000)
    latitude = np.arcsin(rng.uniform(-1.0, 1.0, 20000))
    axis = np.asarray(beam_vector) / np.linalg.norm(beam_vector)
    angle = np.degrees(np.arccos(np.clip(directions(longitude, latitude) @ axis, -1.0, 1.0)))
    inside = np.zeros(len(longitude), dtype=bool)
    for area in areas:
        inside |= area.contains_points(np.column_stack([longitude, latitude]))
    # a margin around the edge, which the polygon follows only to within the spacing of its points
    margin = 0.05 * ang_radius_deg
    assert np.all(inside[angle < ang_radius_deg - margin])
    assert not np.any(inside[angle > ang_radius_deg + margin])


def test_angular_radius_checked():
    with pytest.raises(ValueError, match="between 0 and 180"):
        beam_sky_outline((0.0, 0.0, 1.0), 0.0)
