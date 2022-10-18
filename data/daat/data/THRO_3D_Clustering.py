# Static Clustering Algorithm developed at TH Rosenheim
# &copy; 2020 Dominik Stecher, TH Rosenheim  

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import random

def data_from_clusters(data, cluster, multiplier = 3, keep_ratio = True):
    ''' Data generation from cluster algorithm
    
    Inputs:
    ----------
    data:        dataset to generate synthetic data from
    cluster:     array with dataset id and value cluster number
                 return value of static_clustering() function
    multiplier:
    keep_ratio:
    
    Output:
    ----------
    result:      synthetic data array
    
    Example:
    ----------
    syn_data = data_from_clusters(orginal_data, static_clustering(orginal_data))
    '''
    (cluster_id, samples_per_cluster) = np.unique(cluster, return_counts=True)
    samples_to_generate = samples_per_cluster * multiplier
    
    result = np.zeros((np.sum(samples_to_generate), len(data[0])))
    nins = 0

    for i in range(0, len(cluster_id)):
        pure_cluster = np.zeros((samples_per_cluster[i], len(data[0])))
        ins = 0

        # Collect all samples of cluster
        for j in range(0, len(data)):
            if cluster[j] == cluster_id[i]:
                pure_cluster[ins] = data[j]
                ins += 1

        # Calculate mean and standard deviation
        mean = np.mean(pure_cluster, axis=0)
        std = np.std(pure_cluster, axis=0)

        # Generate new data

        for j in range(0, samples_to_generate[i]):
            for k in range(0, len(data[0])):
                result[nins][k] = random.gauss(mean[k], std[k])
            nins += 1

    return result


#Clustering algorithm. 
# data: Dataset you want clustered
# clusters: Number of Clusters generated
# preassigned_percentage: After selecting the cluster seeds randomly from the dataset, this percentage of data
#                         is assigned to the closest cluster seed to generate the mean and standard deviation
#                         for each cluster
# iterations: Since the initial cluster seeds are picked randomly, multiple iterations are performed to find 
#             the optimal solution similar to other clustering algorithms

# return value: 

def static_clustering(data, clusters = 128, preassigned_percentage = 10, iterations = 5):
    ''' Clustering algorithm.
    
    Inputs:
    ----------
    data: Dataset you want clustered
    clusters: Number of Clusters generated
    preassigned_percentage: After selecting the cluster seeds randomly from the 
                            dataset, this percentage of data is assigned to the 
                            closest cluster seed to generate the mean and 
                            standard deviation for each cluster
    iterations: Since the initial cluster seeds are picked randomly, multiple 
                iterations are performed to find the optimal solution similar 
                to other clustering algorithms
    Output:
    ----------
    result:     numpy array with index = data id and value = best cluster 
    
    Example:
    ----------
    syn_data = data_from_clusters(orginal_data, static_clustering(orgianl_data))
    '''
    assignment = np.zeros((len(data)))
    result = np.zeros((len(data)))
    assignment.fill(-1)
    best_quality = 1e200
    preassigned_samples = int(len(data)*(1/preassigned_percentage))

    for i in range(0, iterations):

        init_c_mean = np.zeros((clusters, len(data[0])))
        init_c_std = np.zeros((clusters, len(data[0])))
        assignment.fill(-1)
        preassigned_centers = np.zeros((clusters, len(data[0])))

        # pick random starting points
        for j in range(0, clusters):
            rnd = random.randint(0, len(data)-1)
            while assignment[rnd] != -1:
                rnd = random.randint(0, len(data) - 1)
            assignment[rnd] = j
            preassigned_centers[j] = data[rnd]

        # find nearest x percent
        distances = np.zeros((clusters, len(data)))
        # calculate distances
        for j in range(0, clusters):
            for k in range(0, len(data)):
                distances[j][k] = np.linalg.norm(data[k] - preassigned_centers[j])
        for j in range(0, len(data)):
            if assignment[j] != -1:
                distances[int(assignment[j])][j] = 1e200
        cluster_seeds = np.zeros((clusters, preassigned_samples, len(data[0])))

        # partially assign data to cluster seeds
        for j in range(0, preassigned_samples):
            for k in range(0, clusters):
                hit = np.where(distances[k] == np.amin(distances[k]))
                cluster_seeds[k][j] = data[hit]
                distances[k][hit] = 1e200

        # calculate mean/std for clusters
        for j in range(0, clusters):

            init_c_mean[j] = np.mean(cluster_seeds[j], axis=0)
            for k in range(0, clusters):
                for l in range(0, preassigned_samples):
                    cluster_seeds[k][l] = cluster_seeds[k][l] - init_c_mean[k]
            init_c_std[j] = np.std(cluster_seeds[j], axis=0)

        # reassign all points
        samples_per_cluster = np.zeros((clusters))
        for j in range(0, len(data)):
            ptc_distance = 1e200
            best_cluster = -1
            for k in range(0, clusters):
                dist = np.sum(np.absolute((data[j] - init_c_mean[k])) / init_c_std[k])
                if dist < ptc_distance:
                    ptc_distance = dist
                    best_cluster = k
            assignment[j] = best_cluster
            samples_per_cluster[best_cluster] += 1

        # calculate homogeneity
        homogeneity = 0
        for j in range(0, clusters):
            pure_cluster = np.zeros((int(samples_per_cluster[j]), len(data[0])))
            ip = 0
            for k in range(0, len(data)):
                if assignment[k] == j:
                    pure_cluster[ip] = data[k]
                    ip += 1
            mean = np.mean(pure_cluster, axis=0)
            for k in range(0, len(pure_cluster)):
                homogeneity += np.sum(np.absolute(pure_cluster[k] - mean))/len(data[0])

        homogeneity /= len(data)
        
        s = ""
        
        if homogeneity <= best_quality:
            result = assignment
            best_quality = homogeneity
            s = "Better result found in "
        s += "Iteration: " + str(i)
        print(s)

    return result


def data_from_clusters_2(data, cluster, n_samples):
    ''' Data generation from cluster algorithm
    
    Inputs:
    ----------
    data:        dataset to generate synthetic data from
    cluster:     array with dataset id and value cluster number
                return value of static_clustering() function
    n_samples:   how many new samples should be generated
    
    Output:
    ----------
    result:      synthetic data array
    
    Example:
    ----------
    syn_data = data_from_clusters(orginal_data, static_clustering(orginal_data))
    '''
    cluster_id, samples_per_cluster = np.unique(cluster, return_counts=True)
    result = np.zeros(( int(n_samples), len(data[0]) ))
    nins = 0
    min_values = np.amin(data, axis=0)
    
    # modified to produce n_samples rather than multiple of input count
    n_samples_per_cluster = int(n_samples / len(cluster_id))
    
    for i in range(0, len(cluster_id)):
        pure_cluster = np.zeros((samples_per_cluster[i], len(data[0])))
        ins = 0
    
        # Collect all samples of cluster
        for j in range(0, len(data)):
            if cluster[j] == cluster_id[i]:
                pure_cluster[ins] = data[j]
                ins += 1
    
        # Calculate mean and standard deviation
        mean = np.mean(pure_cluster, axis=0)
        std = np.std(pure_cluster, axis=0)
    
        # Generate new data
        for j in range(0, n_samples_per_cluster):
            for k in range(0, len(data[0])):
                val = np.random.uniform(mean[k], std[k])
                #val = random.gauss(mean[k], std[k])                
                # changed to return abs() value if value range is positiv
                if min_values[k] >= 0.0 : 
                    val = abs(val)
                result[nins][k] = val
            nins += 1
    
    return result

def static_clustering_2(data, clusters = 128, preassigned_percentage = 10, iterations = 6, verbose = 1):
    ''' Clustering algorithm.
    
    Inputs:
    ----------
    data: Dataset you want clustered
    clusters: Number of Clusters generated
    preassigned_percentage: After selecting the cluster seeds randomly from the
                            dataset, this percentage of data is assigned to the
                            closest cluster seed to generate the mean and
                            standard deviation for each cluster
    iterations: Since the initial cluster seeds are picked randomly, multiple
                iterations are performed to find the optimal solution similar
                to other clustering algorithms
    Output:
    ----------
    result:     numpy array with index = data id and value = best cluster 
    
    Example:
    ----------
    syn_data = data_from_clusters(orginal_data, static_clustering(orgianl_data))
    '''
    assignment = np.zeros((len(data)))
    result = np.zeros((len(data)))
    assignment.fill(-1)
    best_quality = 1e200
    preassigned_samples = int(len(data)*(1/preassigned_percentage))
    
    # if too many clusters are wanted for data size
    if clusters > len(data): 
        print('Not enough data for', clusters, 'cluster.')
        print(len(data))
        clusters = len(data)
    
    for i in range(0, iterations):
        init_c_mean = np.zeros((clusters, len(data[0])))
        init_c_std = np.zeros((clusters, len(data[0])))
        assignment.fill(-1)
        preassigned_centers = np.zeros((clusters, len(data[0])))
        
        # pick random starting points\n",
        for j in range(0, clusters):
            rnd = random.randint(0, len(data)-1)
            while assignment[rnd] != -1:
                rnd = random.randint(0, len(data) - 1)
            assignment[rnd] = j
            preassigned_centers[j] = data[rnd]

        # calculate distances and find nearest x percent
        distances = np.zeros((clusters, len(data)))
    
        for j in range(0, clusters):
            for k in range(0, len(data)): 
                if assignment[k] != -1:
                    distances[j][k] = 1e200
                else:
                    distances[j][k] = np.linalg.norm(data[k] - preassigned_centers[j])

        # partially assign data to cluster seeds (X% der Daten)
        cluster_seeds = np.zeros((clusters, preassigned_samples, len(data[0])))
    
        for j in range(0, preassigned_samples):
            for k in range(0, clusters):
                hit = np.where(distances[k] == np.amin(distances[k]))
                cluster_seeds[k][j] = data[np.amin(hit)] #data[hit]
                distances[k][hit] = 1e200
    
        # calculate mean/std for clusters
        for j in range(0, clusters):
            init_c_mean[j] = np.mean(cluster_seeds[j], axis=0)
            
            for l in range(0, preassigned_samples):
                cluster_seeds[k][l] = cluster_seeds[k][l] - init_c_mean[k]
            
            init_c_std[j] = np.std(cluster_seeds[j], axis=0)
           
        # reassign all points
        samples_per_cluster = np.zeros((clusters))
        for j in range(0, len(data)):

            ptc_distance = 1e200
            best_cluster = -1
            for k in range(0, clusters):
                if 0.0 in init_c_std[k] : 
                    dist = 0.0
                else: 
                    dist = np.sum(np.absolute((data[j] - init_c_mean[k])) / init_c_std[k])
                
                if dist < ptc_distance:
                    ptc_distance = dist
                    best_cluster = k
                    
            assignment[j] = best_cluster
            samples_per_cluster[best_cluster] += 1
            
        # calculate homogeneity
        homogeneity = 0
        for j in range(0, clusters):
            pure_cluster = np.zeros((int(samples_per_cluster[j]), len(data[0])))
            ip = 0
            for k in range(0, len(data)):
                if assignment[k] == j:
                    pure_cluster[ip] = data[k]
                    ip += 1
            mean = np.mean(pure_cluster, axis=0) 
            for k in range(0, len(pure_cluster)):
                homogeneity += np.sum(np.absolute(pure_cluster[k] - mean))/len(data[0])

        homogeneity /= len(data)
        
        s = ""
        
        if homogeneity <= best_quality:
            result = assignment
            best_quality = homogeneity
            s = "Better result found in "
        s += "Iteration: " + str(i)
        if verbose == 1: print(s)
    
    return result

#Synthetic data generation
def line_gen(p, s, u, n, t):
    ret = np.zeros((n, 3))
    for i in range(0, n):
        dev = np.array([random.uniform(1-t, 1+t), random.uniform(1-t, 1+t), random.uniform(1-t, 1+t)])
        ret[i] = np.array((p + random.choice((-1, 1)) * random.uniform(-u, u) * s)) * dev
    return ret


def circle_gen_z(p, r, n, t):
    ret = np.zeros((n, 3))
    for i in range(0, n):
        dev = np.array([random.uniform(-t, t), random.uniform(-t, t), random.uniform(-t, t)])
        ret[i][0] = random.uniform(-r, r)
        ret[i][2] = random.choice((-1, 1)) * (r**2 - ret[i][0]**2)**(1/2)
        #Use + for identical deviation 
        ret[i] = (ret[i] + p) + dev
    return ret

def circle_gen_y(p, r, n, t):
    ret = np.zeros((n, 3))
    for i in range(0, n):
        dev = np.array([random.uniform(1-t, 1+t), random.uniform(1-t, 1+t), random.uniform(1-t, 1+t)])
        ret[i][0] = random.uniform(-r, r)
        ret[i][1] = random.choice((-1, 1)) * (r**2 - ret[i][0]**2)**(1/2)
        ret[i] = (ret[i] + p) * dev
    return ret