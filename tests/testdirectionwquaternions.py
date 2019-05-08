import numpy as np
import quaternion
import time

# with quaternions

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
    # q_polar = (y, theta)
    q_polar = np.quaternion(c_hTheta, 0., s_hTheta,  0.)

    # step 2. phi azimutal angle roation around d
    # q2 = (d, phi)
    q_az = np.quaternion(c_hPhi, d[0]*s_hPhi, d[1]*s_hPhi, d[2]*s_hPhi)

    return quaternion.rotate_vectors((q_az*q_polar), d)


def newdir_n(s_hTheta, c_hTheta, s_hPhi, c_hPhi, x_local, d):
    # step 0. find the axis of polar rotation
    q_phi = np.quaternion(c_hPhi, d[0]*s_hPhi, d[1]*s_hPhi, d[2]*s_hPhi)
    x_rotated = rotate_vector_R(q_phi, x_local)
    n = np.cross(d, x_rotated)
    n = n/ np.linalg.norm(n)

    q_n = np.quaternion(c_hTheta, n[0]*s_hTheta, n[1]*s_hTheta, n[2]*s_hTheta)

    new_dir = rotate_vector_R(q_n, d)

    x_local = rotate_vector_R(q_n, x_local)

    return (new_dir, x_local)
    #return quaternion.rotate_vectors((q_az*q_polar), d)

# with direction cosines
def newdircos_oldMC(sphi, cphi, spsi, cpsi, cxyz):
    # From MCML paper START
    if (abs(cxyz[2]) > 0.99999):
        cxyzp = np.array([sphi * cpsi, sphi * spsi, (cxyz[2]/np.abs(cxyz[2])) * cphi ])
        print 'edge case'
    else:
        dsq = np.sqrt(1.-cxyz[2]*cxyz[2])
        dsqi = 1./dsq
        cxyzp = np.array([ sphi * (cxyz[0] * cxyz[2] * cpsi - cxyz[1] * spsi) * dsqi + cxyz[0] * cphi, \
                sphi * (cxyz[1] * cxyz[2] * cpsi + cxyz[0] * spsi) * dsqi + cxyz[1] * cphi, \
                -sphi * cpsi * dsq + cxyz[2] * cphi ])

    #  From MCML paper END

    # normalize the direction cosines
    dd = 1./np.sqrt(cxyzp.dot(cxyzp))

    cxyzp_norm = cxyzp * dd
    return cxyzp_norm

# with direction cosines
def newdircos_Joy(sphi, cphi, spsi, cpsi, cxyz):
    # From MCML paper START
    A = -1.*cxyz[0]/cxyz[2]
    B = 1./np.sqrt(1.+A**2)
    V1 = A * sphi
    V2 = A * B * sphi
    V3 = cpsi
    V4 = spsi

    cxyzp = np.array([ cxyz[0] * cpsi + V1*V3 + cxyz[1] * V2 * V4, \
                cxyz[1] * cpsi + V4*(cxyz[2] * V1 -  cxyz[0] * V2), \
                cxyz[2] * cpsi + (V2 * V3) - (cxyz[1] * V1 * V4) ])

    #  From MCML paper END

    #print cxyzp
    # normalize the direction cosines
    dd = 1./np.sqrt(cxyzp.dot(cxyzp))

    cxyzp_norm = cxyzp * dd
    return cxyzp_norm

tilt = 20. # degrees
d = np.array([np.sin(np.radians(tilt)), 0.,  np.cos(np.radians(tilt))])

#cxyz = np.array([1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.)])
cxyz = d
#print d
theta = 30. #degrees

stheta = np.sin(np.radians(theta))
ctheta = np.cos(np.radians(theta))
shtheta = np.sin(np.radians(theta)/2.)
chtheta = np.cos(np.radians(theta)/2.)

phi = 10. #degrees

sphi = np.sin(np.radians(phi))
cphi = np.cos(np.radians(phi))
shphi = np.sin(np.radians(phi)/2.)
chphi = np.cos(np.radians(phi)/2.)

#qrot = quaternion.from_spherical_coords(np.radians(phi), np.radians(psi))
#print 'this is just the quaternion from z', qrot

start1_time = time.time()
print 'new direction using quaternions:',newdir(shtheta, chtheta, shphi, chphi, d)
print("--- %s seconds ---" % (time.time() - start1_time))
print

start2_time = time.time()
print 'new direction using quaternions but the right way:', newdir_n(shtheta, chtheta, shphi, chphi, np.array([1., 0., 0.]), d)[0]
print("--- %s seconds ---" % (time.time() - start2_time))
print
print 'new x local', newdir_n(shtheta, chtheta, shphi, chphi, np.array([1., 0., 0.]), d)[1]
print
start3_time = time.time()
print 'new direction using dir cosines:', newdircos_oldMC(stheta, ctheta, sphi, cphi, cxyz)
print("--- %s seconds ---" % (time.time() - start3_time))
print
