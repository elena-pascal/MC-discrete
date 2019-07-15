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


class wrongUpdateOrder(Error):
    ''' Raised when the energy loss is larger than the current energy when calculating scattering angles'''
    pass







# inform the user about what's going wrong




def ElossGTEnergy(e, tableE, eloss, tableW, indxW, type):
    print (' Fatal error:', err)
    print (' in compute_Eloss for in scattering class for', type)
    print (' Value of energy loss larger than half the electron energy.')
    print (' The current energy is:',  e)
    print (' The corresponding energy in the tables is:',  tableE)
    print (' The current energy lost is:',  eloss)
    print (' The neighbouring energy losses in the tables is:',  tableW)
    print (' Stopping.')
    sys.exit()
