from matplotlib import pyplot as plt
import os
import numpy as np
import cv2 # OpenCV

class Calibrator:
    def __init__(self, grid_size=(8,11)):
        self.grid_size = grid_size
        self.homographies = []
        self.real_coords = None
        self.pix_coords = []
        self.K = np.zeros((3,3))
        self.projection_matrices = []
        self.reprojection_errors = []
        self.distortion_parameters = np.array([0,0])
        
    def estimate_homography(self, image)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        return_value, corners = cv2.findChessboardCorners(image, patternSize=self.grid_size)
        assert(return_value)
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        corners = corners.reshape((88,2)).copy()
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001)
        cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)
        A = np.empty((0,9), dtype=float)
        O = np.array([0,0,0]).reshape(1,3)
        square_size = 11/1000 # meters
        real_coords = []
        pix_coords = []
        for index, corner in enumerate(corners):
            Xpix = corner[0] # in pixel
            Ypix = corner[1]
            pix_coords.append(np.array([Xpix, Ypix]))
            grid_size_cv2 = tuple(reversed(self.grid_size)) # OpenCV and Python store matrices differently
            u_index, v_index = np.unravel_index(index, grid_size_cv2) # convert index from linear to 2D (0-based indexing)
            Xmm = u_index * square_size
            Ymm = v_index * square_size
            real_coords.append(np.array([Xmm, Ymm, 0, 1]))
            m = np.array([Xmm, Ymm, 1]).reshape(1,3)
            A = np.vstack((A, np.hstack((m,O,-Xpix*m))))
            A = np.vstack((A, np.hstack((O,m,-Ypix*m))))

        _, _, Vh = np.linalg.svd(A)
        h = Vh.transpose()[:,-1]
        H = h.reshape(3,3)
        return H, np.array(real_coords), np.array(pix_coords)
    
    def estimate_projection_matrix(self, H:np.ndarray, K:np.ndarray)->np.ndarray:
        K_inv = np.linalg.inv(K)
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]
        lambdas = 1/np.linalg.norm(K_inv@h1)
        r1 = lambdas * K_inv@h1
        r2 = lambdas * K_inv@h2
        t = lambdas * K_inv@h3
        t = t.reshape((-1, 1))
        r3 = np.cross(r1, r2)
        R = np.hstack((r1.reshape((-1, 1)), r2.reshape((-1, 1)), r3.reshape((-1, 1))))
        U, _ , V_t = np.linalg.svd(R)
        R = U@V_t
        P = K@np.hstack((R, t))
        return P
    
    def reprojection_error (self, P:np.ndarray, R:np.ndarray, I:np.ndarray):
        epsilon_tot=0
        for i in range(R.shape[0]):
            epsilon = ((np.dot(P[0], R[i]) / np.dot(P[2], R[i]))-I[i][0])**2 + ((np.dot(P[1], R[i]) / np.dot(P[2], R[i]))-I[i][1])**2
            epsilon_tot += epsilon
        return epsilon_tot
    
    def calculate_V(self)->np.ndarray:
        V=np.zeros((0,6))
        for H in self.homographies:
            v11=self._vij_function(H, 0, 0)
            v12=self._vij_function(H, 0, 1)
            v22=self._vij_function(H, 1, 1)
            v=np.array([v12.T, (v11-v22).T])
            V=np.vstack((V, v))
        return V
    
    def estimate_K(self, V:np.ndarray):
        _, _, S_T = np.linalg.svd(V)
        b = S_T.transpose()[:, -1]
        B11, B12, B22, B13, B23, B33 = tuple(b)
        v0 = (B12 * B13 - B11 * B23)/(B11 * B22 - B12 * B12)
        l = B33 - (B13**2 + v0*(B12 * B13 - B11 * B13))/B11
        alpha = np.sqrt(l/B11)
        beta = np.sqrt(l * B11/(B11 * B22 - B12**2))
        gamma = - B12 * (alpha**2) * beta / l
        u0 = gamma * v0 / beta - B13 * (alpha**2) / l
        self.K = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])

    def estimate_distortion(self):
        K = self.K
        skewness = np.arctan(K[0,0]/K[0,1])
        alpha_u, alpha_v = K[0,0], np.abs(K[1,1]*np.sin(skewness))
        u0, v0 = K[0,2], K[1,2]
        A = np.empty((0, 2))
        b = np.empty((0, 1))
        for i,P in enumerate(self.projection_matrices):
            for j,point in enumerate(self.real_coords):
                projected = P@point
                u = projected[0]/projected[2]
                v = projected[1]/projected[2]
                rd2 = ((u-u0)/alpha_u)**2+((v-v0)/alpha_v)**2
                first_row = np.array([(u-u0)*rd2, (u-u0)*(rd2)**2])
                second_row = np.array([(v-v0)*rd2, (v-v0)*(rd2)**2])
                A = np.vstack((A, first_row, second_row))
                uh = self.pix_coords[i][j,0]
                vh = self.pix_coords[i][j,1]
                b = np.vstack((b, np.array([uh-u]), np.array([vh-v])))
        distortion_params = np.linalg.inv(A.T@A)@A.T@b
        self.distortion_parameters = distortion_params
        return distortion_params

    
    def fit(self, images:list[np.ndarray], radial_distortion=False, iterative=False):
        for image in images:
            H, real_coords, pix_coords = self.estimate_homography(image)
            if self.real_coords is None:
                self.real_coords = real_coords
            self.pix_coords.append(pix_coords)
            self.homographies.append(H)
        V = self.calculate_V()
        self.estimate_K(V)        
        for H in self.homographies:
            P = self.estimate_projection_matrix(H, self.K)
            self.projection_matrices.append(P)
        print(radial_distortion)
        if radial_distortion is True:
            self.estimate_distortion()
            print('eseguito')
        if iterative:
            pass
        for i,P in enumerate(self.projection_matrices):
            error = self.reprojection_error(P, self.real_coords, self.pix_coords[i])
            self.reprojection_errors.append(error)

    def _vij_function (self, H, i, j):
        v=np.zeros(6)
        v[0]=H[0][i]*H[0][j]
        v[1]=H[0][i]*H[1][j]+H[1][i]*H[0][j]
        v[2]=H[1][i]*H[1][j]
        v[3]=H[2][i]*H[0][j]+H[0][i]*H[2][j]
        v[4]=H[2][i]*H[1][j]+H[1][i]*H[2][j]
        v[5]=H[2][i]*H[2][j]
        return v
