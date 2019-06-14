from timing_decorator import time_md5_comparison

USE_DECORATOR = [False]

@time_md5_comparison(USE_DECORATOR)
def md5_comparison(a, b):
    if a==b:
        return True
    else:
        return False

print md5_comparison(3, 4)  # prints nothing
print 'end of first call'
#print USE_DECORATOR[0]
USE_DECORATOR[0] = True
md5_comparison(5, 6)  # prints timing info
