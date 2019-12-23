
from functools import wraps
import time

# from scimath.units.api import has_units
# def has_units_fact(use_decorator):
#     # a decorator factory that toggles @has_units on or off
#     #print 'inside factory', use_decorator
#     def decorator(function):
#         #print 'inside decorator', use_decorator
#         @wraps(function)
#         def wrapped(*args, **kwargs):
#             #print 'inside wrapped', use_decorator
#             if use_decorator[0]: # use the has_units decorator
#                 result = has_units(function(*args, **kwargs))
#                 print '**yes'
#             else: # just return the unitless function
#                 result = function(*args, **kwargs)
#             return result
#
#         return wrapped
#
#     return decorator




def time_this(enabled):
    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if enabled[0]:
                t1 = time.time()
                result = function(*args, **kwargs)
                t2 = time.time()
                print(str(function.__name__)+"  "+ str("%.6f " %(t2 - t1)))
            else:
                result = function(*args, **kwargs)

            return result

        return wrapped

    return decorator
