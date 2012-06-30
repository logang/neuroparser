import math

def ExponentialMap(minimum, maximum):
    diff = math.log(maximum) - math.log(minimum)
    minimum = math.log(minimum)
    fractToValue = lambda x:math.exp(diff*x + minimum)
    if diff:
        valueToFract = lambda x:(math.log(x) - minimum)/diff
    else:
        valueToFract = lambda x:0.5
    return fractToValue, valueToFract

def LinearMap(minimum, maximum):
    """
    A basic linear map for SliderWidget
    """
    diff = maximum-minimum
    fractToValue = lambda x: x*(maximum-minimum) + minimum
    if diff == 0:
        valueToFract = lambda x: 0.0
    else:
        valueToFract = lambda x: (x-minimum)/(maximum-minimum)
    return fractToValue, valueToFract

def LinearIntMap(minimum, maximum):
    """
    A linear integer map for SliderWidget
    """
    num_values = maximum-minimum+1
    fractToValue = lambda x: int(x*(num_values-1)) + minimum
    valueToFract = lambda x: (1.0*(x-minimum) + 0.5) / num_values 
    return fractToValue, valueToFract

def SquareRootMap(minimum, maximum):
    """
    A basic inverse quadratic map for SliderWidget
    """
    diff = maximum-minimum
    def fractToValue(x):
        return math.sqrt(x)*diff + minimum
    if diff == 0:
        valueToFract = lambda x: 0.5
    else:
        def valueToFract(x):
            return math.pow((x-minimum)/diff,2.0)
    return fractToValue, valueToFract
