# calculate scattering direction using quaternions


def rotate_vector_Lqv(q, v):
    '''
    v is the vector to be rotated
    q is the rotation as a quaternion
    '''
    rotated_v = ( 2 * q.real*q.real - 1. ) * v + 2*( (q.imag.dot(v))*q.imag + q.real*np.cross(q.imag, v) )
    return rotated_v

def rotate_vector_R(q, v):
    '''
    v is the vector to be rotated
    q is the rotation as a quaternion
    '''
    rotated_v =  v + np.cross(2.*q.imag, (np.cross(q.imag, v) + q.real*v))
    return rotated_v


def newdir(shphi, chphi, shpsi, chpsi, d):
    # step 1. phi polar angle roation around y
    # q_polar = (y=[0,1,0], phi)
    q_polar = np.quaternion(chphi, 0., shphi,  0.)

    # step 2. psi azimutal angle roation around
    q_az = np.quaternion(chpsi, d[0]*shpsi, d[1]*shpsi, d[2]*shpsi)
    return rotate_vector_R((q_az*q_polar), d)
