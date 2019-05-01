class Error(Exception):
    ''' Bare class for exceptions'''
    pass

# define some errors for the m values in integrals
class mTooSmall(Error):
    ''' Raised when the iteration step m is negative'''
    pass

class mTooLarge(Error):
    ''' Raised when the iteration step m becomes too large'''
    pass


# define some errors for the limits of path lengths
class lTooSmall(Error):
    ''' Raised when the path length calcualted is too small'''
    pass

class lTooLarge(Error):
    ''' Raised when the path length calcualted is too large'''
    pass


# define some errors for the limits of energy loss
class E_lossTooSmall(Error):
    ''' Raised when the energy loss calcualted is too small'''
    pass

class E_lossTooLarge(Error):
    ''' Raised when the energy loss calcualted is too large'''
    pass


# define some errors for the quaternions losing normalisation
class q_polNotNormal(Error):
    ''' Raised when the azimuthal quaternion is not normalised'''
    pass

class q_azNotNormal(Error):
    ''' Raised when the polar quaternion is not normalised'''
    pass
