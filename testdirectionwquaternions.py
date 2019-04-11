import numpy as np
import quaternion

# with quaternions

def rotate_vector(v, q):
    '''
    v is the vector to be rotated
    q is the rotation as a quaternion
    '''
    rotated_v = ( q.real**2 - abs(q.norm())**2 )*v + 2*( (q.imag.dot(v))*q.imag + q.real*np.cross(q.imag, v) )
    return rotated_v

def newdirquat(shphi, chphi, shpsi, chpsi, d):
    # step 1. phi polar angle roation around y
    # q_polar = (y=[0,1,0], phi)
    q_polar = np.quaternion(chphi, 0., shphi, 0.)
    #print q_polar
    # step 2. psi azimutal angle roation around
    # q2 = (d, psi)
    q_az = np.quaternion(chpsi, d[0]*shpsi, d[1]*shpsi, d[2]*shpsi)
    #print q_az
    #return rotate_vector(d, q_polar*q_az)
    return quaternion.rotate_vectors(q_polar*q_az, d)

# with direction cosines
def newdircos_oldMC(sphi, cphi, spsi, cpsi, cxyz):
    # From MCML paper START
    if (abs(cxyz[2]) > 0.99999):
        cxyzp = ([sphi * cpsi, sphi * spsi, (cxyz[2]/abs(cxyz[2])) * cphi ])
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
    if (abs(cxyz[2]) > 0.99999):
        cxyzp = ([sphi * cpsi, sphi * spsi, (cxyz[2]/abs(cxyz[2])) * cphi ])
        print 'edge case'
    else:
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

    print cxyzp
    # normalize the direction cosines
    #dd = 1./np.sqrt(cxyzp.dot(cxyzp))

    #cxyzp_norm = cxyzp * dd
    return cxyzp

d = np.array([0., 0., 1.])
cxyz = np.array([0., 0., 1.])

phi = 60 # degrees
sphi = np.sqrt(3.)/2
cphi = 0.5

shphi = 0.5
chphi = np.sqrt(3.)/2

psi = 90 # degrees
spsi = 1.
cpsi = 0.

shpsi = np.sqrt(2.)/2
chpsi = np.sqrt(2.)/2

print 'new direction using quaternions:', newdirquat(shphi, chphi, shpsi, chpsi, d)
print 'new direction using dir cosines Joy:', newdircos_Joy(sphi, cphi, spsi, cpsi, cxyz)
print 'new direction using dir cosines:', newdircos_oldMC(sphi, cphi, spsi, cpsi, cxyz)
