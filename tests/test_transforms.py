import numpy
from snakeML import numpy_transformations


def test_transforms():
    mat=numpy.array([[2, 4],[5,6]])
    print(numpy_transformations.mcol(mat))

test_transforms()