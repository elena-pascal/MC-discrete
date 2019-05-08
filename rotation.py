# calculate scattering direction using quaternions
import numpy as np
import quaternion
import sys

from errors import q_azNotNormal, q_polNotNormal

def rotate_vector_Lqv(q, v):
    '''
    v is the vector to be rotated
    q is the rotation as a quaternion
    '''
    rotated_v = ( 2. * q.real*q.real - 1. ) * v + 2.*( (q.imag.dot(v))*q.imag + q.real*np.cross(q.imag, v) )
    return rotated_v

def rotate_vector_R(q, v):
    '''
    v is the vector to be rotated
    q is the rotation as a quaternion
    '''
    rotated_v =  v + np.cross(2.*q.imag, (np.cross(q.imag, v) + q.real*v))
    return rotated_v


#def newdir(s_hTheta, c_hTheta, s_hPhi, c_hPhi, x_local, d):
    # step 0. find the axis of polar rotation
#    q_phi = np.quaternion(c_hPhi, d[0]*s_hPhi, d[1]*s_hPhi, d[2]*s_hPhi)
#    x_rotated = rotate_vector_R(q_phi, x_local)
#    n = np.cross(d, x_rotated)
#    n = n/ np.linalg.norm(n)
#
#    q_n = np.quaternion(c_hTheta, n[0]*s_hTheta, n[1]*s_hTheta, n[2]*s_hTheta)
#
#    #new_dir = rotate_vector_R((q_az*q_polar), d)
#    new_dir = rotate_vector_R(q_n, d)
#    x_local = rotate_vector_R(q_n, x_local)
#    #print x_local
#    return (new_dir, x_local)

def newdir(s_hTheta, c_hTheta, s_hPhi, c_hPhi, y_local, d):
    # step 1. theta polar angle roation around y
    # q_polar = (y, theta)
    q_polar = np.quaternion(c_hTheta, y_local[0]*s_hTheta, y_local[1]*s_hTheta,  y_local[2]*s_hTheta)

    # step 2. phi azimutal angle roation around d
    # q2 = (d, phi)
    q_az = np.quaternion(c_hPhi, d[0]*s_hPhi, d[1]*s_hPhi, d[2]*s_hPhi)

    # step 3. total rotation quaternions
    q_total = q_az*q_polar

    return (quaternion.rotate_vectors(q_total, d), quaternion.rotate_vectors(q_total, y_local))
