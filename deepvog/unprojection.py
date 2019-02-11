import numpy as np


# The whole unprojection algorithm is invented by Safaee-Rad et al. 1992, see https://ieeexplore.ieee.org/document/163786
# This python script is a re-implementation of Safaee-Rad et al.'s works


def gen_cone_co(alpha, beta, gamma, a_prime, h_prime, b_prime, g_prime, f_prime, d_prime):
    gamma_square = np.power(gamma,2)
    a = gamma_square * a_prime
    b = gamma_square * b_prime
    c = a_prime * np.power(alpha,2) + 2 * h_prime * alpha * beta + b_prime * np.power(beta,2) + 2 * g_prime * alpha + 2 * f_prime * beta + d_prime
    d = gamma_square * d_prime
    f = -gamma * (b_prime * beta + h_prime * alpha +f_prime)
    g = -gamma * (h_prime * beta + a_prime * alpha + g_prime)
    h = gamma_square * h_prime
    u = gamma_square * g_prime
    v = gamma_square * f_prime
    w = -gamma * (f_prime * beta + g_prime * alpha + d_prime)
    return a,b,c,d,f,g,h,u,v,w

'''
Safaee-Rad, 1992 (8)

'''
def gen_rotmat_co(lamb, a,b,g,f,h):
    t1 = (b-lamb)*g - f*h
    t2 = (a - lamb)*f - g*h
    t3 = -(a-lamb)*(t1/t2)/g - (h/g)
    m = 1/(np.sqrt(1+np.power((t1/t2),2)+np.power(t3,2)))
    l = (t1/t2)*m
    n = t3*m
    return l, m, n
'''
Safaee-Rad, 1992 (12), (27)-(33)

'''
def gen_lmn(lamb1, lamb2, lamb3):
    if lamb1 < lamb2:
        l = 0
        m_pos = np.sqrt((lamb2-lamb1)/(lamb2-lamb3))
        m_neg = -m_pos
        n = np.sqrt((lamb1-lamb3)/(lamb2-lamb3))
        return [l, l], [m_pos, m_neg], [n, n]
    elif lamb1 > lamb2:
        l_pos = np.sqrt((lamb1-lamb2)/(lamb1-lamb3))
        l_neg = -l_pos
        n = np.sqrt((lamb2-lamb3)/(lamb1-lamb3))
        m = 0
        return [l_pos, l_neg], [m, m], [n, n]
    elif lamb1 == lamb2:
        n = 1
        m = 0
        l = 0
        return [l,l], [m,m], [n,n]
    else:
        
        logging.warning("Failure to generate l,m,n. None's are returned")
        return None, None, None
def calT3(l, m ,n ):
    lm_sqrt = np.sqrt((l**2)+(m**2))
    T3 = np.array([-m/lm_sqrt, -(l*n)/lm_sqrt, l, 0,
                       l/lm_sqrt, -(m*n)/lm_sqrt, m, 0,
                       0, lm_sqrt, n, 0,
                       0, 0, 0, 1]).reshape(4,4)
    return T3
def calABCD(T3, lamb1, lamb2, lamb3):
    li, mi, ni = T3[0:3,0], T3[0:3,1], T3[0:3,2]
    lamb_array = np.array([lamb1, lamb2, lamb3])
    A = np.dot(np.power(li,2), lamb_array)
    B = np.sum(li*ni*lamb_array)
    C = np.sum(mi*ni*lamb_array)
    D = np.dot(np.power(ni,2), lamb_array)
    return A,B,C,D
def calXYZ_perfect(A,B,C,D, r):
    
    Z = (A*r)/np.sqrt((B**2)+(C**2)-A*D)
    X = (-B/A)*Z
    Y = (-C/A)*Z
    center = np.array([X,Y,Z,1]).reshape(4,1)
    return center
    
    
    
def check_parallel(v1, v2):
    a = np.dot(v1.T, v2)
    b = np.linalg.norm(v1) * np.linalg.norm(v2)
    radian = np.arccos(a/b).squeeze()
    return np.rad2deg(radian)
def convert_ell_to_general(xc,yc, w,h,radian):
    A = (w**2)*(np.sin(radian)**2) + (h**2) * (np.cos(radian)**2)
    B = 2 * ((h**2) - (w**2)) * np.sin(radian) * np.cos(radian)
    C = (w**2)*(np.cos(radian)**2) + (h**2)*(np.sin(radian)**2)
    D = -2*A*xc - B*yc
    E = -B*xc - 2*C*yc
    F = A*(xc**2) + B*xc*yc + C*(yc**2) - (w**2)*(h**2)
    return A,B,C,D,E,F


def unprojectGazePositions(vertex, ell_co, radius = None):
    """
    This function generates (1)directions of unprojected pupil disk (gaze vector) 
    and (2) position of the pupil disk, with an assumed radius of the pupil disk

    Args:
        vectex (list or tuple): list with 3 elements of x, y, z coordinates of the camera with respect to the image frame
        ell_co (list or tuple): list of 6 coefficients of a generalised/expanded ellipse equations at the image frame
            A*(x**2) + B*x*y + C*(y**2) + D*x + E*y + F = 0 (from https://en.wikipedia.org/wiki/Ellipse#General_ellipse)
        
    Returns:
        Positive Norm of pupil disk from camera frame
        Negative Norm of pupil disk from camera frame
        Positive Norm of pupil disk from canonical frame
        negative Norm of pupil disk from canonical frame
    """
    
    # Coefficients of the general ellipse equation
    A,B,C,D,E,F = [x for x in ell_co]
    # Vertex (Point of the camera)
    alpha, beta, gamma = [x for x in vertex]
    
    # Ellipse parameter at image frame (z_c = +20) with respect to the camera frame
    a_prime = A
    h_prime = B/2
    b_prime = C
    g_prime = D/2
    f_prime = E/2
    d_prime = F
    # Coefficients of the Cone at the image frame
    a,b,c,d,f,g,h,u,v,w = gen_cone_co(alpha, beta, gamma, a_prime, h_prime, b_prime, g_prime, f_prime, d_prime)
    # Safaee-Rad, 1992 (10)
    lamb_co1 = 1
    lamb_co2 = -(a+b+c)
    lamb_co3 = (b*c + c*a + a*b - np.power(f,2) - np.power(g,2) - np.power(h,2))
    lamb_co4 = -(a*b*c + 2*f*g*h - a*np.power(f,2) - b*np.power(g,2) - c*np.power(h,2))
    lamb1, lamb2, lamb3 = np.roots([lamb_co1, lamb_co2, lamb_co3, lamb_co4])
    # generate Normal vector at the canonical frame
    
    l, m, n = gen_lmn(lamb1,lamb2,lamb3)
    norm_cano_pos = np.array([l[0],m[0],n[0],1]).reshape(4,1)
    norm_cano_neg = np.array([l[1],m[1],n[1],1]).reshape(4,1)
    
    # T1 Rotational Transformation to the camera fream
    l1, m1, n1 = gen_rotmat_co(lamb1, a,b,g,f,h)
    l2, m2, n2 = gen_rotmat_co(lamb2, a,b,g,f,h)
    l3, m3, n3 = gen_rotmat_co(lamb3, a,b,g,f,h)
    T1 = np.array([l1,l2,l3,0 ,m1, m2, m3,0, n1, n2, n3,0, 0,0,0,1]).reshape(4,4)
    li, mi, ni = T1[0,0:3], T1[1,0:3], T1[2,0:3]
    if np.cross(li,mi).dot(ni) < 0:
        li = -li
        mi = -mi
        ni = -ni
    T1[0,0:3], T1[1,0:3], T1[2,0:3] = li, mi, ni
    norm_cam_pos = np.dot(T1,  norm_cano_pos)
    norm_cam_neg = np.dot(T1, norm_cano_neg)

    # Calculating T2
    T2 = np.eye(4)
    T2[0:3,3] = -(u*li+v*mi+w*ni)/np.array([lamb1, lamb2, lamb3])
    # Calculating T3
    T3_pos = calT3(l[0], m[0], n[0])
    T3_neg = calT3(l[1], m[1], n[1])
    # calculate ABCD
    A_pos, B_pos, C_pos, D_pos = calABCD(T3_pos, lamb1, lamb2, lamb3)
    A_neg, B_neg, C_neg, D_neg = calABCD(T3_neg, lamb1, lamb2, lamb3)
    # Calculating T0
    T0 = np.eye(4)

    T0[2,3] = -gamma # -gamma = -(vertex[2]) = -(-focal_length) = + focal_length
    # Calculating center position with respect to the perfect frame
    center_pos = calXYZ_perfect(A_pos, B_pos, C_pos, D_pos, radius)
    center_neg = calXYZ_perfect(A_neg, B_neg, C_neg, D_neg, radius)
    # From perfect frame to camera frame
    true_center_pos = np.matmul(T0,np.matmul(T1,np.matmul(T2,np.matmul(T3_pos,center_pos))))
    if true_center_pos[2] <0:
        center_pos[0:3] = -center_pos[0:3]
        true_center_pos = np.matmul(T0,np.matmul(T1,np.matmul(T2,np.matmul(T3_pos,center_pos))))
    true_center_neg = np.matmul(T0,np.matmul(T1,np.matmul(T2,np.matmul(T3_neg,center_neg))))
    if true_center_neg[2] <0:
        center_neg[0:3] = -center_neg[0:3]
        true_center_neg = np.matmul(T0,np.matmul(T1,np.matmul(T2,np.matmul(T3_neg,center_neg))))

    return norm_cam_pos[0:3], norm_cam_neg[0:3], true_center_pos[0:3], true_center_neg[0:3]

def reproject(vec_3d, focal_length, batch_mode= False):
    # vec_3d = (3,1) numpy array: Coordinates of the 3D unprojected object in CAMERA frame
    # vec_3d can also be (3,), but not (1,3)
    if batch_mode == False:
        vec_2d = (focal_length*vec_3d[0:2])/vec_3d[2]
        
    else:
        # converting vec_3d ~ (m,3) to vec_2d~(m,2)
        vec_2d = (focal_length*(vec_3d[:,0:2]))/vec_3d[:,[2]]
    return vec_2d
    
def reverse_reproject(vec_2d, z, focal_length):
    # Scale the x,y in a reverse manner of reproject() function,
    # when you unproject the reprojected coordinate.
    vec_2d_scaled = (vec_2d*z)/focal_length
    return vec_2d_scaled

# Illustration of the example from Safaee-Rad's paper
if __name__ == "__main__":
    pass
    
    