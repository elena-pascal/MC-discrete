# calculate scattering direction using quaternions
import numpy as np
import quaternion

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


def newdir(s_hTheta, c_hTheta, s_hPhi, c_hPhi, d):
    # step 1. theta polar angle roation around y
    # q_polar = (y=[0,1,0], half theta), theta = [0, pi]
    q_polar = np.quaternion(c_hTheta, 0., s_hTheta,  0.)

    # step 2. phi azimutal angle roation around d
    # q_az = (d=[dx, dy, dz], half phi), phi = [0, 2pi]
    q_az = np.quaternion(c_hPhi, d[0]*s_hPhi, d[1]*s_hPhi, d[2]*s_hPhi)

    return rotate_vector_R((q_az*q_polar), d)
