import numpy as np


class NoIntersectionError(Exception):
    pass

# vector (m,n) , m = number of examples, n = dimensionality
# a = coordinates of the vector
# n = orientation of the vector
def intersect(a, n):
    # default normalisation of vectors n
    n = n/np.linalg.norm(n, axis = 1, keepdims=True)
    num_lines = a.shape[0]
    dim = a.shape[1]
    I = np.eye(dim)
    R_sum = 0
    q_sum = 0
    for i in range(num_lines):    
        R = I - np.matmul(n[i].reshape(dim,1), n[i].reshape(1,dim))

        q = np.matmul(R,a[i].reshape(dim,1))
        q_sum = q_sum + q
        R_sum = R_sum + R
    p = np.matmul(np.linalg.inv(R_sum),q_sum)
    return p

def calc_distance(a,n, p):
    num_lines = a.shape[0]
    dim = a.shape[1]
    I = np.eye(dim)
    D_sum = 0
    for i in range(num_lines):
        D_1 = (a[i].reshape(dim,1) - p.reshape(dim,1)).T
        D_2 = I - np.matmul(n[i].reshape(dim,1), n[i].reshape(1,dim))
        D_3 = D_1.T
        D = np.matmul(np.matmul(D_1,D_2),D_3)
        D_sum = D_sum + D
    D_sum = D_sum/num_lines
    return D_sum

def fit_ransac(a,n, max_iters = 2000, samples_to_fit = 20, min_distance = 2000):
    num_lines = a.shape[0]
    
    best_model = None
    best_distance = min_distance
    for i in range(max_iters):
        # print("\rRANSAC: Currently {0}".format(i), flush=True)
        sampling_index = np.random.choice(num_lines, size = samples_to_fit, replace=False)
        a_sampled = a[sampling_index,:]
        n_sampled = n[sampling_index,:]
        model_sampled = intersect(a_sampled, n_sampled)
        sampled_distance = calc_distance( a,n, model_sampled)
        # print(sampled_distance)
        if sampled_distance > min_distance:
            continue
        else:
            if sampled_distance < best_distance:
                best_model = model_sampled
                best_distance = sampled_distance
    # if best_model is None:
    #     best_model = model_sampled
    return best_model

def line_sphere_intersect(c, r, o, l):
    # c = numpy array (3,1). Centre of the eyeball
    # r = scaler. Radius of the eyeball
    # o = numpy array (3,1). Origin of the line
    # l = numpy array (3,1). Directional unit vector of the line
    # return [d1, d2] : auxilary variables of the parametrised line x = o + dl
    # the closer one to the camera is chosen
    l = l/np.linalg.norm(l)
    delta = np.square(np.dot(l.T,(o-c))) - np.dot((o-c).T,(o-c)) + np.square(r)
    if delta < 0:
        raise NoIntersectionError
    else:
        d1 = -np.dot(l.T,(o-c)) + np.sqrt(delta)
        d2 = -np.dot(l.T,(o-c)) - np.sqrt(delta)
    return [d1,d2]
    

#%%
if __name__ == "__main__":
    
    pass

    