import numpy as np
from copy import deepcopy
import pywren
import numpywren
from numpywren import matrix, matrix_utils 
from numpywren.matrix_init import shard_matrix
import make_coding_function
import time


lam = 0.01
n_procs=60  
num_parity_blocks=6 

pwex = pywren.lambda_executor()

def conjugate_gradient_method(H, g, num_parity_blocks, coding_length, n_iters=20, thres):
    H_coded = make_coding_function.code_2D(H, num_parity_blocks, thres=thres)
    d = np.zeros((len(g), 1))
    r_old = g.reshape(-1, 1) - coded_mat_vec_mul(H_coded, d, num_parity_blocks, coding_length)
    p = r_old
    for _ in range(n_iters):
        Hp = coded_mat_vec_mul(H_coded, p, num_parity_blocks, coding_length)
        a = r_old.T.dot(r_old) / (p.T.dot(Hp))
        d += a.squeeze()*p
        r_new = r_old - a*Hp
        if np.linalg.norm(r_new) < 1e-6:
            return dict
        b = r_new.T.dot(r_new) /  r_old.T.dot(r_old)
        p = r_new + b*p
        r_old = r_new
    return d
    
    
def sketched_sqrt_hessian(id, X, gamma, hashes, flips, n_features, N):
    x, y = id[0], id[1]
    A = X.get_block(0, x)
    a = gamma.get_block(0, 0)
    m, n = A.shape
    sqrt_H = np.zeros(A.shape)
    for i in range(n):
        sqrt_H[:,i] = np.multiply(A[:,i], np.squeeze(a))

    sqrt_H = sqrt_H.T
    hash_local = hashes[y, :]
    flip_local = flips[y, :]
    sketch_block = np.zeros((m, n_features))
    for i in range(m):
        sketch_block[:, hash_local[i]] += flip_local[i]*sqrt_H[:,i]
    return sketch_block/np.sqrt(N), id


def approx_hessian(iterable, X_s3_train_2, gamma, N, n_features, n_samples, lam, thres=0.05):
    hashes = np.random.randint(0, n_features, size=(N, n_samples))
    flips = np.random.choice([-1, 1], size=(N, n_samples))
    futures = pwex.map(lambda x: sketched_sqrt_hessian(x, X_s3_train_2, gamma, hashes, flips, n_features, N), iterable)

    sketch_sqrt_H = np.zeros((n_features, n_features*N))
    x_shard_size = X_s3_train_2.shard_sizes[1]
    y_shard_size = n_features
    not_dones = list(range(len(futures)))
    iterable_not_done = deepcopy(iterable)
    while len(not_dones) >= thres*len(futures):
        fs_dones, fs_not_dones = pywren.wait(futures,2)
        for i, future in enumerate(futures):
            if future in fs_dones and i in not_dones:
                try:
                    x_ord, y_ord = future.result()[1]
                    sketch_sqrt_H[x_ord*x_shard_size:(x_ord+1)*x_shard_size, y_ord*y_shard_size:(y_ord+1)*y_shard_size] = future.result()[0] 
                    not_dones.remove(i)
                    iterable_not_done.remove((x_ord,y_ord))
                except Exception as e:
                    print(e)
                    pass

    return sketch_sqrt_H.dot(sketch_sqrt_H.T) / n_samples + lam*np.eye(sketch_sqrt_H.shape[0]) 


def oversketched_newton(X_train, X_test, y_train, y_test, N, n_iters, n_iters_cg, thres=0.9):
    n_samples, n_features = X_train.shape
    n_samples_test = X_test.shape[0]

    X_s3_train_1, X_s3_train_2, X_s3_test, y_s3_test, y_s3_conv = prepare_data(X_train, X_test, y_train, y_test)

    X_train_coded = make_coding_function.code_2D(X_train, num_parity_blocks, thres=thres)
    X_train_T_coded = make_coding_function.code_2D(X_s3_train_2.T, num_parity_blocks, thres=thres)

    coding_length = int(np.ceil(len(X_s3_train_1._block_idxs(0))/num_parity_blocks))

    y_train = y_train.reshape((n_samples, 1))
    y_test = y_test.reshape((n_samples_test, 1))

    w_loc = np.zeros((n_features, 1))
    w, beta, gamma = initialize_parameters(n_samples, n_features)
    w.put_block(w_loc, 0, 0)

    iterable = [(x,y) for x in X_s3_train_2._block_idxs(1) for y in range(N)]

    alpha_train = np.zeros((n_samples, 1))
    alpha_test = np.zeros((n_samples_test, 1))
    training_loss = [np.mean(np.log(1+np.exp(-np.multiply(y_train, alpha))))]
    testing_loss  = [np.mean(np.log(1+np.exp(-np.multiply(y_test, alpha_test))))]

    iterations_sec = [0]
    for _ in range(n_iters):     
        start = time.time()  

        beta_loc = np.divide(y_train, np.exp(np.multiply(alpha, y_train)) + 1)
        beta.put_block(beta_loc, 0, 0)     

        g = coded_mat_vec_mul(X_train_T_coded, beta, num_parity_blocks, coding_length)
        g = (-1/n_samples)*g + lam*w_loc
        
        a = np.divide(np.exp(np.multiply(alpha, y_train)), np.square(np.exp(np.multiply(alpha, y_train)) + 1))
        gamma.put_block(np.sqrt(a), 0, 0)  
        
        H = approx_hessian(iterable, gamma, N, n_features, n_samples, lam, thres=0.05)
        step = conjugate_gradient_method(H, g, num_parity_blocks, coding_length, n_iters_cg, thres)
        w_loc = np.subtract(w_loc , step)
        
        w.put_block(w_loc, 0, 0)

        sec = time.time() - start
        iterations_sec.append(sec)

        alpha = coded_mat_vec_mul(X_train_coded, w, num_parity_blocks, coding_length)
        alpha_test = X_test.dot(w_loc)
        
        training_loss.append(np.mean(np.log(1+np.exp(-np.multiply(y_train,alpha)))))
        testing_loss.append(np.mean(np.log(1+np.exp(-np.multiply(y_test, alpha_test)))))
    return iterations_sec, training_loss, testing_loss
    
    
def prepare_data(X_train, X_test, y_train, y_test):
    n_samples, n_features = X_train.shape
    n_samples_test = X_test.shape[0]

    X_s3_train_1 = matrix.BigMatrix(
        "features_{0}_{1}_{2}".format(n_samples, n_features, n_procs), 
        shape=(n_samples, n_features), 
        shard_sizes=(n_samples//n_procs, n_features), 
        write_header=True
    )
    shard_matrix(X_s3_train_1, X_train, overwrite=True)

    X_s3_train_2 = matrix.BigMatrix(
        "features_{0}_{1}_{2}".format(n_samples, n_features, n_procs), 
        shape=(n_samples, n_features), 
        shard_sizes=(n_samples, int(np.ceil(n_features/n_procs))), 
        write_header=True
    )
    shard_matrix(X_s3_train_2, X_train, overwrite=True)

    X_s3_test = matrix.BigMatrix(
        "features_test_{0}_{1}".format(n_samples_test, n_features), 
        shape=(n_samples_test, n_features), 
        shard_sizes=(n_samples_test, n_features), 
        write_header=True
    )
    shard_matrix(X_s3_test, X_test, overwrite=True)

    y_s3_train = matrix.BigMatrix(
        "y_train_{0}_{1}".format(n_samples, n_procs), 
        shape=(n_samples,), 
        shard_sizes=(n_samples//n_procs,), 
        write_header=True
    )
    shard_matrix(y_s3_train, y_train, overwrite=True)

    y_s3_test = matrix.BigMatrix(
        "y_test_{0}".format(n_samples_test), 
        shape=(n_samples_test,), 
        shard_sizes=(n_samples_test,), write_header=True
    )
    shard_matrix(y_s3_test, y_test, overwrite=True)


def initialize_parameters(n_samples, n_features):
    w = matrix.BigMatrix(
        "w_t_{0}".format(n_features), 
        shape=(n_features, 1), 
        shard_sizes=(n_features, 1),
        write_header=True
    )

    beta = matrix.BigMatrix(
        "beta_t_{0}".format(n_samples), 
        shape=(n_samples,1), 
        shard_sizes=(n_samples,1), 
        autosqueeze=False, 
        write_header=True
    )

    gamma = matrix.BigMatrix(
        "gamma_t_{0}".format(n_samples), 
        shape=(n_samples,1), 
        shard_sizes=(n_samples,1), 
        autosqueeze=False, 
        write_header=True
    )

    return w, beta, gamma
