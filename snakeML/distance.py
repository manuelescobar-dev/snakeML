def euclidean_distance_objects(obj1, obj2):
    """
    Computes the euclidean distance between two objects with attributes x and y

    Parameters
    ----------
    obj1: object 1 (with attributes x and y)
    obj2: object 2 (with attributes x and y)
    """
    return ((obj1.x-obj2.x)**2 + (obj1.y-obj2.y)**2)**0.5

def euclidean_distance(x1,x2,y1,y2):
    """
    Computes the euclidean distance between two points (x1,y1) and (x2,y2)

    Parameters
    ----------
    x1: x coordinate of point 1
    x2: x coordinate of point 2
    y1: y coordinate of point 1
    y2: y coordinate of point 2
    """
    return ((x1-x2)**2 + (y1-y2)**2)**0.5