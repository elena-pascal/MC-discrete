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

def newdirquat(shphi, chphi, shpsi, chpsi, d):
    # step 1. phi polar angle roation around y
    # q_polar = (y=[0,1,0], phi)
    q_polar = np.quaternion(chphi, 0., shphi,  0.)
    #print 'q polar', q_polar
    #print q_polar/np.sqrt(q_polar.w**2 + q_polar.x**2 + q_polar.y**2 + q_polar.z**2)
    # step 2. psi azimutal angle roation around
    # q2 = (d, psi)
    #print 'after polar rotation', quaternion.rotate_vectors(q_polar, d)

    q_az = np.quaternion(chpsi, d[0]*shpsi, d[1]*shpsi, d[2]*shpsi)
    #print 'after both rotations', quaternion.rotate_vectors(q_az, quaternion.rotate_vectors(q_polar, d))
    #print 'q azimuthal', q_az
    #print 'as spherical coords:', quaternion.as_spherical_coords(q_polar*q_az)
    return rotate_vector_R((q_az*q_polar), d)
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

d = np.array([2., 0., 2.])
d = d/np.sqrt(d.dot(d))
#cxyz = np.array([1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.)])
cxyz = d
#print d
phi = 45. #degrees

sphi = np.sin(np.radians(phi))
cphi = np.cos(np.radians(phi))
shphi = np.sin(np.radians(phi)/2.)
chphi = np.cos(np.radians(phi)/2.)

psi = 20. #degrees

spsi = np.sin(np.radians(psi))
cpsi = np.cos(np.radians(psi))
shpsi = np.sin(np.radians(psi)/2.)
chpsi = np.cos(np.radians(psi)/2.)

#qrot = quaternion.from_spherical_coords(np.radians(phi), np.radians(psi))
#print 'this is just the quaternion from z', qrot

start1_time = time.time()
print 'new direction using quaternions:',newdirquat(shphi, chphi, shpsi, chpsi, d)
print("--- %s seconds ---" % (time.time() - start1_time))
print

start2_time = time.time()
print 'new direction using dir cosines Joy:', newdircos_Joy(sphi, cphi, spsi, cpsi, cxyz)
print("--- %s seconds ---" % (time.time() - start2_time))
print
start3_time = time.time()
print 'new direction using dir cosines:', newdircos_oldMC(sphi, cphi, spsi, cpsi, cxyz)
print("--- %s seconds ---" % (time.time() - start3_time))
print
