from __future__ import division
import numpy as np
import numpy.polynomial.polynomial as poly
from scipy import constants
from scipy.optimize import newton

def y_deflection(initial_trajectory, flight_length, speed):
    """
    The vertical displacement of an object moving in a parabolic path.

    Parameters
    ----------
    initial_trajectory : float
        Initial trajectory of the object (degrees).  A positive angle lies
        above the x-axis.
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
    traj_rad = np.radians(initial_trajectory)
    ret = flight_length * np.tan(traj_rad)
    ret -= (constants.g * flight_length ** 2
            / 2. / speed ** 2 / np.cos(traj_rad) ** 2.)
    return ret


def elevation(initial_trajectory, flight_length, speed):
    """
    The immediate inclination of an object moving in a parabolic path
    """
    traj_rad = np.radians(initial_trajectory)
    ret = np.tan(traj_rad)
    ret -= constants.g * flight_length / (speed * np.cos(traj_rad)) ** 2.
    return np.degrees(np.arctan(ret))


def find_trajectory(theta, flight_length, speed):
    """
    Find the initial trajectory of an object that has to pass through a certain
    point on a parabolic path

    Parameters
    ----------
    theta : float
        The angle between the x-axis and the point (degrees). A positive angle
        lies above the x-axis.
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


def parabola_line_intersection_point(initial_trajectory, speed, theta,
                                     flight_length, omega):
    """
    Find the intersection point of the parabolic motion path and a line.
    The line is specified by an angle and a point through which the line
    goes.

    Parameters
    ----------
    initial_trajectory : float
        The initial trajectory of the object (degrees).
    speed : float
        The initial speed of the object (m/s).
    theta : float
        The declination of a point on the line from the origin (degrees).
    flight_length : float
        The horizontal component of the distance of a point on the line from
        the origin (m).
    omega : float
        The arctan of the gradient of the line (degrees) going through that
        point (degrees).  i.e. np.tan(np.radians(omega)) is the gradient of
        the line.

    Returns
    -------
    intersect_x, intersect_y, distance : float, float, float
        Co-ordinates of intersection point of parabola and line, and distance
        from the intersection to the point used to specify the line.
    """
    omega_rad = np.radians(omega)
    theta_rad = np.radians(theta)
    traj_rad = np.radians(initial_trajectory)

    # equation of line
    a1 = flight_length * (np.tan(theta_rad) - np.tan(omega_rad))
    b1 = np.tan(omega_rad)

    # equation of parabola
    a2 = 0.
    b2 = np.tan(traj_rad)
    c2 = -constants.g / 2. / (speed * np.cos(traj_rad)) ** 2.

    intersection_x = np.max(poly.polyroots([a2 - a1, b2 - b1, c2]))
    intersection_y = a1 + intersection_x * b1
    distance = ((intersection_x - flight_length) ** 2
                + (intersection_y - flight_length * np.tan(theta_rad)) ** 2)
    return intersection_x, intersection_y, np.sqrt(distance)
