##### Modules
import numpy as np 
import pandas as pd

def project_points_onto_axes(points, x_point1, x_point2, y_point1, y_point2):
    """
    Projects a 2D numpy array of n-dimensional points onto orthogonal axes defined by two pairs of n-dimensional points,
    while ensuring that the axes are orthogonal and the intersection point of these two axes is the origin.

    Parameters:
    -----------
    points : numpy array
        A 2D numpy array with shape (N, D), where N is the number of points and D is the number of dimensions.
    x_point1 : numpy array
        The first n-dimensional point that defines the x-axis.
    x_point2 : numpy array
        The second n-dimensional point that defines the x-axis.
    y_point1 : numpy array
        The first n-dimensional point that defines the y-axis.
    y_point2 : numpy array
        The second n-dimensional point that defines the y-axis.

    Returns:
    --------
    numpy array
        A 1D numpy array with shape (N,), where each element contains the magnitude of the n-dimensional point
        projected onto the x-axis defined by the input points.
    numpy array
        A 1D numpy array with shape (N,), where each element contains the magnitude of the n-dimensional point
        projected onto the y-axis defined by the input points.
    """
    # Compute the unit vectors along the axes
    x_axis_vector = (x_point2 - x_point1) / 2
    y_axis_vector = (y_point2 - y_point1) / 2
    x_axis_vector = x_axis_vector / np.linalg.norm(x_axis_vector, ord=2)
    y_axis_vector = y_axis_vector / np.linalg.norm(y_axis_vector, ord=2)
    # Now length of the vector is 1
    cos = np.dot(x_axis_vector,y_axis_vector)
    # Project each point onto the x-axis and y-axis
    x_projection = []
    y_projection = []
    x_dist = []
    y_dist = []
    x_middle = (x_point1 + x_point2) / 2
    y_middle = (y_point1 + y_point2) / 2
    x1x = np.dot(x_point1 - x_middle , x_axis_vector) 
    x2x = np.dot(x_point2 - x_middle, x_axis_vector) 
    y1y = np.dot(y_point1 - y_middle, y_axis_vector) 
    y2y = np.dot(y_point2 - y_middle, y_axis_vector) 
    x1y = np.dot(x_point1 - y_middle, y_axis_vector) 
    x2y = np.dot(x_point2 - y_middle, y_axis_vector) 
    y1x = np.dot(y_point1 - x_middle, x_axis_vector) 
    y2x = np.dot(y_point2 - x_middle, x_axis_vector) 
    x1xtrue = x1x - x1y*cos
    x1ytrue = x1y - x1x*cos
    x2xtrue = x2x - x2y*cos
    x2ytrue = x2y - x2x*cos
    y1xtrue = y1x - y1y*cos
    y1ytrue = y1y - y1x*cos
    y2xtrue = y2x - y2y*cos
    y2ytrue = y2y - y2x*cos
    xorigin, yorigin = line_intersection((x1xtrue,x2xtrue,y1xtrue,y2xtrue), (x1ytrue,x2ytrue,y1ytrue,y2ytrue))
    x_negative_scale = np.abs(x1xtrue - xorigin)
    x_positive_scale = np.abs(x2xtrue - xorigin)
    y_negative_scale = np.abs(y1ytrue - yorigin)
    y_positive_scale = np.abs(y2ytrue - yorigin)
    for point in points:
        x_proj = np.dot(point - x_middle , x_axis_vector)  
        y_proj = np.dot(point - y_middle , y_axis_vector) 
        true_y = (y_proj - x_proj*cos) - yorigin
        true_x = (x_proj - y_proj*cos) - xorigin
        if true_x < 0:
            true_x = true_x / x_negative_scale
        else:
            true_x = true_x / x_positive_scale
        if true_y < 0:
            true_y = true_y / y_negative_scale
        else:
            true_y = true_y / y_positive_scale
        x_projection.append(true_x)
        y_projection.append(true_y)
        x_dist.append(np.linalg.norm(point - x_proj*x_axis_vector, ord=2))
        y_dist.append(np.linalg.norm(point - y_proj*y_axis_vector, ord=2))

    # Return the magnitudes of the projections as numpy arrays
    return np.array(x_projection), np.array(y_projection), np.array(x_dist), np.array(y_dist)

def line_intersection(x_pts, y_pts):  
    line1 = ((x_pts[0], y_pts[0]), (x_pts[1], y_pts[1]))
    line2 = ((x_pts[2], y_pts[2]), (x_pts[3], y_pts[3]))
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def load_emotions():
    emotions_dict = {"PV" : ["Happy", "Pleased", "Delighted", "Excited", "Satisfied"],
                        "NV" : ["Miserable", "Frustrated", "Sad", "Depressed", "Afraid"],
                        "HA" : ["Astonished", "Alarmed", "Angry", "Afraid", "Excited"],
                        "LA" : ["Tired", "Sleepy", "Calm", "Satisfied", "Depressed"]}
    return emotions_dict