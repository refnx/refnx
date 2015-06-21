from __future__ import division
import numpy as np
from numpy.polynomial import Polynomial
from scipy import constants
from scipy.optimize import newton

def y_deflection(initial_trajectory, flight_length, speed):
    """
    The vertical displacement of an object moving in a parabolic path.

    Parameters
    ----------
    initial_trajectory : float
        The initial angular trajectory of the path (degrees), measured from
        the x-axis. A positive angle is in an anticlockwise direction.
    flight_length : float
        The horizontal component of the distance of the object from the origin
        (m).
    speed : float
        The initial speed of the object (m/s)

    Returns
    -------
    displacement : float
        Vertical displacement of the object (m).
    """
    eqn = parabola(initial_trajectory, speed)
    return eqn(flight_length)


def elevation(initial_trajectory, flight_length, speed):
    """
    Angle between x-axis and tangent of flight path at a given horizontal
    distance from the origin.

    Parameters
    ----------
    initial_trajectory : float
        The initial angular trajectory of the path (degrees), measured from
        the x-axis. A positive angle is in an anticlockwise direction.
    flight_length : float
        The horizontal component of the distance of a point on the line from
        the origin (m).
    speed : float
        The initial speed of the object (m/s)

    Returns
    -------
    elevation : float
        The direction in which the object is currently moving (degrees). The
        angle is measured relative to the x-axis, with a positive angle in an
        anticlockwise direction.
    """
    eqn = parabola(initial_trajectory, speed)
    dydx = eqn.deriv()
    return np.degrees(np.arctan(dydx(flight_length)))


def find_trajectory(theta, flight_length, speed):
    """
    Find the initial trajectory of an object that has to pass through a certain
    point on a parabolic path.  This point is specified by the horizontal distance
    component and an angle (polar coordinates).

    Parameters
    ----------
    theta : float
        The angle between the x-axis and the point (degrees). A positive angle
        is in an anticlockwise direction.
    flight_length : float
        The horizontal component of the distance of a point on the line from
        the origin (m).
    speed : float
        The initial speed of the object (m/s)

    Returns
    -------
    trajectory : float
        Initial trajectory of the object (degrees).  A positive angle lies
        above the x-axis.
    """
    theta_rad = np.radians(theta)
    vertical_deflection_vertex = np.tan(theta_rad) * flight_length

    # now find trajectory that will put object through defined point
    def traj(trajectory):
        return (vertical_deflection_vertex
                - y_deflection(trajectory, flight_length, speed))

    trajectory = newton(traj, 0)
    return trajectory


def parabola(initial_trajectory, speed):
    """
    Find the quadratic form of the parabolic path

    Parameters
    ----------
    initial_trajectory : float
        The initial angular trajectory of the path (degrees), measured from
        the x-axis. A positive angle is in an anticlockwise direction.
    speed : float
        The initial speed of the object (m/s)

    Returns
    -------
    eqn : np.polynomial.Polynomial object
        Equation of parabolic path
    """
    traj_rad = np.radians(initial_trajectory)
    eqn = Polynomial([0,
                      np.tan(traj_rad),
                      -constants.g / 2. / (speed * np.cos(traj_rad)) ** 2.])
    return eqn


def parabola_line_intersection_point(initial_trajectory, flight_length, speed,
                                     theta, omega):
    """
    Find the intersection point of the parabolic motion path and a line.
    The line is specified by an angle and a point through which the line
    goes.

    Parameters
    ----------
    initial_trajectory : float
        The initial trajectory of the object (degrees). A positive angle is in
        an anticlockwise direction.
    flight_length : float
        The horizontal component of the distance of a point on the line from
        the origin (m).
    speed : float
        The initial speed of the object (m/s).
    theta : float
        The declination of a point on the line from the origin (degrees). A
        positive angle is in an anticlockwise direction.
    omega : float
        Included angle between the line and line drawn between the point on the
        line and the origin.

    Returns
    -------
    intersect_x, intersect_y, x_prime : float, float, float
        Co-ordinates of intersection point of parabola and line, and distance
        from the intersection to the point used to specify the line.
    """
    omega_rad = np.radians(omega)
    theta_rad = np.radians(theta)

    # omega_prime is the angle between the line and the x-axis. It tells us
    # about the gradient of the line.
    omega_prime = omega_rad + theta_rad
    line_gradient = np.tan(omega_prime)

    # equation of line
    line_eqn = Polynomial([(flight_length
                            * (np.tan(theta_rad) - line_gradient)),
                           line_gradient])

    # equation of parabola
    parab_eqn = parabola(initial_trajectory, speed)

    # find the intersection
    diff = parab_eqn - line_eqn
    intersection_x = np.max(diff.roots())

    # find y location at intersection
    intersection_y = line_eqn(intersection_x)

    # distance between parabola-line intersection and 'point on line'
    distance = ((intersection_x - flight_length) ** 2
                + (intersection_y - flight_length * np.tan(theta_rad)) ** 2)
    x_prime = np.sqrt(distance)
    return intersection_x, intersection_y, x_prime


def find_location_theta(speeds, locations, dist1, dist2):
    """
    Many projectiles are emitted from the origin, over a range of trajectories
    and with a range of speeds. All the projectiles pass through a common point
    located a horizontal distance, `dist1` away from the origin. They then go
    on to pass through a plane which is a horizontal distance, `dist2`, away
    from the origin.  The projectiles pass through this plane at a range of
    y-deflections, `locations`, which is determined by their initial speed and
    trajectory. This function finds the angle, `theta` which the common point
    makes with the x-axis. It also finds the y-deflection (on the detection
    plane)that a particle travelling at infinite speed would have.

    Parameters
    ----------
    speeds : array_like
        Speeds at which the projectiles are emitted from the origin (m/s)
    locations : array_like
        Locations on the detection plane where the projectiles pass through.

    Returns
    -------
    location, theta : float
    """
    pass
