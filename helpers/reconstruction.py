import cv2
import os
import numpy as np
import pickle
import sys
import scipy.io as io
import glob
from scipy import optimize
from tqdm import tqdm
from helpers.multiface import fc_predictor
from helpers.avatars import serialize
from helpers.hephaestus import hephaestus_bindings as hephaestus
from helpers import transform

def _procrustes(X, Y, scaling=True, reflection='best'):
    n, m = X.shape
    ny, my = Y.shape
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY
    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()
    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}
    return d, Z, tform

class NMFCRenderer:
    def __init__(self, args):
        self.args = args

        MM_inpath ='files/all_all_all.mat'
        multiface_path = 'files/A3'
        self.handler = fc_predictor.Handler([multiface_path], 70, 192, args.gpu_id)

        # load sampling indices
        with open('files/sampler1035ver.pkl','rb') as f:
            sampler = pickle.load(f)
        idxs = sampler['idxs']

        # Load 3DMM
        with open('files/exp_30.dat','rb') as f:
            exp_30 = serialize.deserialize_binary_to_morphable_model(f.read())
        self.m_fit = exp_30['mean_points'].reshape((-1, 3))[idxs].astype(np.float32)
        
        MM = io.loadmat(MM_inpath)['fmod']
        self.id_basis = MM['id_basis'][0][0]
        self.exp_basis = MM['exp_basis'][0][0]
        self.mean_shape = MM['mean'][0][0]

        self.M_fit = self.mean_shape.reshape((-1, 3))[idxs]
        Bas = np.concatenate([self.id_basis, self.exp_basis], axis=1)
        Bas_use_3Darr = Bas.T.reshape((Bas.shape[1], -1, 3)).transpose((2, 1, 0))
        Bas_fit = Bas_use_3Darr[:, idxs, :]
        self.Bas_fit = Bas_fit.transpose((2, 1, 0)).reshape((Bas.shape[1], -1)).T

        # initialize hephaestus renderer
        self.width = 256  # NMFC width hardcoded
        self.height = 256  # NMFC height hardcoded
        shaderDir = 'helpers/shaders'   # needs to point to the directory with the hephaestus shaders
        hephaestus.init_system(self.width, self.height, shaderDir)
        hephaestus.set_clear_color(0, 0, 0, 0)

        # create a model from the mean mesh of the 3DMM
        self.model = hephaestus.create_NMFC(exp_30['mean_points'], exp_30['mean_indices'])
        hephaestus.setup_model(self.model)

    def fit_3DMM(self, points, Wbound_Cid=.8, Wbound_Cexp=1.5):
        # Compute optmisation bounds
        num_id = self.id_basis.shape[1]
        num_exp = self.exp_basis.shape[1]
        UBcoefs = np.vstack((Wbound_Cid * np.ones((num_id, 1)), 
                             Wbound_Cexp * np.ones((num_exp,1))))
        Bcoefs = np.hstack((-UBcoefs, UBcoefs))

        # Align points with mean shape
        _, points_aligned, tform = _procrustes(self.M_fit, points, reflection=False)
        b = points_aligned.ravel() - self.M_fit.ravel()

        coefs = optimize.lsq_linear(self.Bas_fit, b, bounds=(Bcoefs[:, 0], Bcoefs[:, 1]), method='trf',
                                    tol=1e-10, lsq_solver=None, lsmr_tol=None, max_iter=None, verbose=0)
        return coefs['x']

    def reconstruct(self, images):
        # Values to return
        cam_params = [] 
        id_params = []
        exp_params = []
        landmarks5 = []
        success = True

        handler_ret_prev = None
        n_consecutive_fails = 0
        # Perform 3D face reconstruction for each given frame.
        print('Running face reconstruction')
        for image in tqdm(images):
            if isinstance(image, str):
                # Read image
                frame = cv2.imread(image)
                if frame is None:
                    print('Failed to read %s' % image)
                    success = False
                    break
            else:
                # If we are given images, convert them to BGR
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Run multiface detector
            handler_ret = self.handler.get(frame)

            # Check if dense landmarks were found and only one face exists in the image.
            if len(handler_ret) == 2:
                # Face(s) found in frame.
                n_consecutive_fails = 0
                landmarks, lands5 = handler_ret[0], handler_ret[1]
                if len(landmarks) > 1:
                    print('More than one faces were found')
                    landmarks, lands5 = landmarks[0:1], lands5[0:1]
            else:
                # Face not found in frame.
                n_consecutive_fails += 1
                print('Failed to find a face (%d times in a row)' % n_consecutive_fails)
                if handler_ret_prev is None or n_consecutive_fails > 5:
                    success = False
                    break
                else:
                    # Recover using previous landmarks
                    handler_ret = handler_ret_prev

            # Perform fitting.
            pos_lms = landmarks[0][:-68].astype(np.float32)
            shape = pos_lms.copy() * np.array([1, -1, -1], dtype=np.float32) # landmark mesh is in left-handed system
            coefs = self.fit_3DMM(shape)
            Pinv = transform.estimate_affine_matrix_3d23d(self.m_fit, pos_lms).astype(np.float32)

            # Gather results.
            cam_params.append(transform.P2sRt(Pinv)) # Scale, Rotation, Translation
            id_params.append(coefs[:157])            # Identity coefficients
            exp_params.append(coefs[157:])           # Expression coefficients
            landmarks5.append(lands5[0])             # Five facial landmarks
            handler_ret_prev = handler_ret

        # Return
        return success, cam_params, id_params, exp_params, landmarks5

    def computeNMFCs(self, cam_params, id_params, exp_params, return_RGB=False):
        nmfcs = []
        print('Computing NMFCs')
        for cam_param, id_param, exp_param in tqdm(zip(cam_params, id_params, exp_params), total=len(cam_params)):
            # Get Scale, Rotation, Translation
            S, R, T = cam_param

            # Compute face without pose.
            faceAll = self.mean_shape.ravel() + np.matmul(self.id_basis, id_param).ravel() + exp_param.dot(self.exp_basis.T)

            # Compute face with pose.
            T = (T / S).reshape(3,1)
            posed_face3d = R.dot(faceAll.reshape(-1, 3).T) + T

            # Use hephaestus to generate the NMFC image.
            hephaestus.update_positions(self.model, posed_face3d.astype(np.float32).T.ravel())

            # setup orthographic projection and place the camera
            viewportWidth = self.width / S
            viewportHeight = self.height / S

            # seems the viewport is inverted for Vulkan, handle this by inverting the ortho projection
            hephaestus.set_orthographics_projection(self.model, viewportWidth * 0.5, -viewportWidth * 0.5,
                                                    -viewportHeight * 0.5, viewportHeight * 0.5, -10, 10)

            # set the cameara to look at the center of the mesh
            target = hephaestus.vec4(viewportWidth * 0.5, viewportHeight * 0.5, 0, 1)
            camera = hephaestus.vec4(viewportWidth * 0.5, viewportHeight * 0.5, -3, 1)
            hephaestus.set_camera_lookat(self.model, camera, target)

            # Render NMFC
            data, channels, width, height = hephaestus.render_NMFC(self.model)

            data3D = data.reshape((height, width, channels))
            data3D = data3D[:,:,0:3]
            if not return_RGB:
                data3D = data3D[..., ::-1]
            nmfcs.append(data3D)
        return nmfcs

    def clear(self):
        # clean up
        hephaestus.clear_system()
