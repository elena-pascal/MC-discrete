from scimath.units.api import has_units

def has_units_fact(use_decorator = False):
    # a decorator factory that toggles @has_units on or off
    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if use_decorator: # use the has_units decorator
                decorator = has_units(function(*args, **kwargs))
            else: # just return the unitless function
                result = function(*args, **kwargs)
            return result

        return wrapped

    return decorator
