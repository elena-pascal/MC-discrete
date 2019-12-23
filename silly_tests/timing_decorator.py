from functools import wraps
import time

def time_md5_comparison(enabled):
    def decorator(function):
        #print 'inside decorator', enabled
        @wraps(function)
        def wrapped(*args, **kwargs):
            #print 'inside wrapped', enabled[0]
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
