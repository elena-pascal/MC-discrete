# calculate scattering direction using quaternions
import numpy as np
from math import cos, sin
import quaternion


from MC.errors import q_azNotNormal, q_polNotNormal

def rotate_vector_Lqv(q, v):
    '''
    v is the vector to be rotated
    q is the rotation as a quaternion
    '''
    return ( 2. * q.real*q.real - 1. ) * v + 2.*( (q.imag.dot(v))*q.imag + q.real*np.cross(q.imag, v) )


def rotate_vector_R(q, v):
    '''
    v is the vector to be rotated
    q is the rotation as a quaternion
    '''
    return  v + np.cross(2.*q.imag, (np.cross(q.imag, v) + q.real*v))



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

    # step 3. total rotation quaternion
    q_total = q_az*q_polar

    return quaternion.rotate_vectors(q_total, (d, y_local))


# with direction cosines
def newdircos_oldMC(sphi, cphi, spsi, cpsi, cxyz):
    # From MCML paper START
    if (abs(cxyz[2]) > 0.99999):
        cxyzp = np.array([sphi * cpsi, sphi * spsi, (cxyz[2]/np.abs(cxyz[2])) * cphi ])
        print ('edge case')
    else:
        dsq = np.sqrt(1.-cxyz[2]*cxyz[2])
        dsqi = 1./dsq
        cxyzp = np.array([ sphi * (cxyz[0] * cxyz[2] * cpsi - cxyz[1] * spsi) * dsqi + cxyz[0] * cphi, \
                sphi * (cxyz[1] * cxyz[2] * cpsi + cxyz[0] * spsi) * dsqi + cxyz[1] * cphi, \
                -sphi * cpsi * dsq + cxyz[2] * cphi ])

    #  From MCML paper END

    # normalize the direction cosines
    dd = 1./np.sqrt(cxyzp.dot(cxyzp))

    return  cxyzp * dd



def projOnDetector(xyz_exit, alpha, xy_PC, L):
    '''
    calculate the projection on the detector of a given escape direction

    xyz_array = list of escape direction unit vectors in sampe frame
    alpha = angle in degrees that brings the detector frame coincidental with the sample frame
            by a CW rotation around x-axis

    returns list of x,y projection on the detector
    '''

    # let t be the translation vector from detector frame origin to the sample frame origin
    t = np.array([xy_PC[0], xy_PC[1], L])

    # let q_SD be the rotation that brings in coincidence the detector frame with the sample frame
    # q_SD = (x, alpha)
    c_hAlpha = cos(np.radians(alpha)/2.)
    s_hAlpha = (1. - c_hAlpha**2)**0.5

    q_SD = np.quaternion(c_hAlpha, s_hAlpha, 0., 0.)

    # the plane equaiton of the detector in the detector frame is z=0
    # so the x,y components of the projections are just the two components of the
    # translated dir vector, obviously
    return [(quaternion.rotate_vectors(q_SD.conj(), one_dir) - t)[0:2]   for one_dir in xyz_exit]



#dir = [np.array([1., 0., 1.]), np.array([1., 1.5, 1.])]
#xy_PC = np.array([4.2, 12.])
#print projOnDetector(dir, 70, xy_PC, 1258.)
