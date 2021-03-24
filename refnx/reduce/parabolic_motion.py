import numpy as np
from numpy.polynomial import Polynomial
from scipy import constants, integrate


@np.vectorize
def y_deflection(initial_trajectory, speed, flight_length):
    """
    The vertical displacement of an object moving in a parabolic path.

    Parameters
    ----------
    initial_trajectory : float
        The initial angular trajectory of the path (degrees), measured from
        the x-axis. A positive angle is in an anticlockwise direction.
    speed : float
        The initial speed of the object (m/s)
    flight_length : float
        The horizontal component of the distance of the object from the origin
        (m).

    Returns
    -------
    displacement : float
        Vertical displacement of the object (m).
    """
    eqn = parabola(initial_trajectory, speed)
    return eqn(flight_length)


def elevation(initial_trajectory, speed, flight_length):
    """
    Angle between x-axis and tangent of flight path at a given horizontal
    distance from the origin.

    Parameters
    ----------
    initial_trajectory : float
        The initial angular trajectory of the path (degrees), measured from
        the x-axis. A positive angle is in an anticlockwise direction.
    speed : float
        The initial speed of the object (m/s)
    flight_length : float
        The horizontal component of the distance of a point on the line from
        the origin (m).

    Returns
    -------
    elevation : float
        The direction in which the object is currently moving (degrees). The
        angle is measured relative to the x-axis, with a positive angle in an
        anticlockwise direction.
    """
    # eqn = parabola(initial_trajectory, speed)
    # dydx = eqn.deriv()
    # return np.degrees(np.arctan(dydx(flight_length)))
    traj_rad = np.radians(initial_trajectory)
    # o_0 = 0
    o_1 = np.tan(traj_rad)
    o_2 = -constants.g / 2.0 / (speed * np.cos(traj_rad)) ** 2.0

    # y = o_0 + o_1*x + o_2*x**2
    # need to work out derivative of y, dy/dx.
    dydx = o_1 + 2 * o_2 * flight_length
    return np.degrees(np.arctan(dydx))


def find_trajectory(flight_length, theta, speed):
    """
    Find the initial trajectory of an object that has to pass through a certain
    point on a parabolic path.  This point is specified by the horizontal
    distance component and an angle (polar coordinates).

    Parameters
    ----------
    flight_length : float
        The horizontal component of the distance of a point on the line from
        the origin (m).
    theta : float
        The angle between the x-axis and the point (degrees). A positive angle
        is in an anticlockwise direction.
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

    # # now find trajectory that will put object through defined point
    # def traj(trajectory):
    #     return (vertical_deflection_vertex -
    #             y_deflection(trajectory, speed, flight_length))
    #
    # trajectory = newton(traj, 0)
    # return trajectory

    # https://en.wikipedia.org/wiki/Trajectory_of_a_projectile
    x = flight_length
    y = vertical_deflection_vertex
    v = speed
    g = constants.g

    # num0 = (v ** 2 + np.sqrt(v ** 4 - g * (g * x ** 2 + 2 * y * v ** 2)))
    num1 = v ** 2 - np.sqrt(v ** 4 - g * (g * x ** 2 + 2 * y * v ** 2))
    # there are two trajectories that could hit the target
    # we only need one branch.
    # alpha1 = np.arctan2(num0, g * x)
    alpha2 = np.arctan2(num1, g * x)

    return np.degrees(alpha2)


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
    eqn = Polynomial(
        [
            0,
            np.tan(traj_rad),
            -constants.g / 2.0 / (speed * np.cos(traj_rad)) ** 2.0,
        ]
    )
    return eqn


@np.vectorize
def parabola_line_intersection_point(
    flight_length, theta, initial_trajectory, speed, omega
):
    """
    Find the intersection point of the parabolic motion path and a line.
    The line is specified by an angle and a point through which the line
    goes.

    Parameters
    ----------
    flight_length : float
        The horizontal component of the distance of a point on the line from
        the origin (m).
    theta : float
        The declination of a point on the line from the origin (degrees). A
        positive angle is in an anticlockwise direction.
    initial_trajectory : float
        The initial trajectory of the object (degrees). A positive angle is in
        an anticlockwise direction.
    speed : float
        The initial speed of the object (m/s).
    omega : float
        Included angle between the line and line drawn between the point on the
        line and the origin.

    Returns
    -------
    intersect_x, intersect_y, x_prime, elev : float, float, float, float
        Co-ordinates of intersection point of parabola and line, distance
        from the intersection to the point used to specify the line, and
        elevation at the intersection point
    """
    omega_rad = np.radians(omega)
    theta_rad = np.radians(theta)

    # omega_prime is the angle between the line and the x-axis. It tells us
    # about the gradient of the line.
    omega_prime = omega_rad + theta_rad
    line_gradient = np.tan(omega_prime)

    # equation of line
    line_eqn = Polynomial(
        [(flight_length * (np.tan(theta_rad) - line_gradient)), line_gradient]
    )

    # equation of parabola
    parab_eqn = parabola(initial_trajectory, speed)

    # find the intersection
    diff = parab_eqn - line_eqn
    intersection_x = np.max(diff.roots())

    # find y location at intersection
    intersection_y = line_eqn(intersection_x)

    # distance between parabola-line intersection and 'point on line'
    distance = (intersection_x - flight_length) ** 2 + (
        intersection_y - flight_length * np.tan(theta_rad)
    ) ** 2
    x_prime = np.sqrt(distance)

    if intersection_x < flight_length:
        x_prime *= -1.0

    # calculate the elevation
    elev = elevation(initial_trajectory, speed, flight_length)

    return intersection_x, intersection_y, x_prime, elev


def arc_length(p, a, b):
    """
    Calculates the arc length of a Polynomial

    Parameters
    ----------
    p: np.polynomial.Polynomial
    a: float
        Lower limit of arc
    b: float
        Upper limit of arc

    Returns
    -------
    length: float
        arc length of polynomial
    """
    p_prime = p.deriv()

    def kernel(x):
        return np.sqrt(1 + p_prime(x) ** 2)

    return integrate.quad(kernel, a, b)[0]
