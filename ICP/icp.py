import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys

def best_fit_transform(data_A, data_B, use_labels):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    # assert A.shape == B.shape

    #Only use xyz coordinates for translation
    if use_labels:
        A = data_A[:, 0:3]
        B = data_B[:, 0:3]
        m = 3
    else:
        A = data_A
        B = data_B
        # get number of dimensions
        m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # print(centroid_A, centroid_B)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    # assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1, metric='euclidean')
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def remove_median(distances, indices):

    
    error = np.mean(distances)

    #always takes 50 % of the data unless the error becomes small
    percentage_data_to_take = max(((1-error)/(1+error))**3, 0.50)

    # other_poss = [round(max(((1-error)/(1+error))**2, 0.25)*100,3), round(max(((1-error)/(1+error))**4, 0.25)*100,3)]

    #Gets the indices of distances when sorted from low to high
    order = np.argsort(distances)

    #Get the percentage to taken
    N_to_take = int(percentage_data_to_take*len(order))
    indices_to_take = order[:N_to_take]

    
    reduced_distances = distances[indices_to_take]
    reduced_indices = indices[indices_to_take]

    return reduced_distances, reduced_indices, indices_to_take, percentage_data_to_take#, other_poss   



def icp(A, B, use_labels=False, init_pose=None, max_iterations=20, tolerance=0.0001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    # assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        reduced_distances, reduced_indices, src_indices, percentage_taken = remove_median(distances, indices)

        # sys.stdout.write("Taking {}% of the data! Should I take {}%, {}%?\r".format(round(percentage_taken*100, 3), other_poss[0], other_poss[1]))
        # sys.stdout.flush()

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,src_indices].T, dst[:m,reduced_indices].T, use_labels)

        # update the current source
        #If we use colors we need to update using only the xyz coordinates
        if use_labels:
            xyz = np.ones((4,src.shape[1]))
            xyz[0:3,:] = src[0:3,:]
            xyz = np.dot(T, xyz)
            src[0:3,:] = xyz[0:3,:].copy()
        else:
            src = np.dot(T, src)

        mean_error = np.mean(distances)
        # check error
        if np.abs(mean_error) < tolerance:
            break
    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T, use_labels)
    print("")

    return T, distances, i, round(mean_error,5)
