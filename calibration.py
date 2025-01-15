import numpy as np
import cv2 # OpenCV
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

class Calibrator:
    def __init__(self, grid_size=(8,11), square_size=11/1000):
        self.grid_size = grid_size
        self.square_size = square_size
        self.reset()

    def reset(self):
        self.homographies = []
        self.real_coords = None
        self.pix_coords = []
        self.K = np.zeros((3,3))
        self.projection_matrices = []
        self.reprojection_errors = []
        self.distortion_parameters = np.array([0,0])
        self.rotations = []
        self.translations = []
        self._no_of_images = 0

        
    def estimate_homography(self, image)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        return_value, corners = cv2.findChessboardCornersSB(gray, patternSize=self.grid_size)
        assert(return_value)
        corners = corners.reshape((self.grid_size[0]*self.grid_size[1],2)).copy()
        A = np.empty((0,9), dtype=float)
        O = np.array([0,0,0]).reshape(1,3)
        square_size = self.square_size # meters
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
    
    def estimate_projection_matrix(self, H:np.ndarray, K:np.ndarray)->tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return P, R, t
    
    def reprojection_error (self, P:np.ndarray, R:np.ndarray, I:np.ndarray, radial:bool):
        epsilon_tot=0
        for i in range(R.shape[0]):
            u = (np.dot(P[0], R[i]) / np.dot(P[2], R[i]))
            v = (np.dot(P[1], R[i]) / np.dot(P[2], R[i]))
            if radial is True:
                K = self.K
                k1 = self.distortion_parameters[0]
                k2 = self.distortion_parameters[1]
                skewness = np.arctan(K[0,0]/K[0,1])
                alpha_u, alpha_v = K[0,0], np.abs(K[1,1]*np.sin(skewness))
                u0, v0 = K[0,2], K[1,2]
                rd2 = ((u-u0)/alpha_u)**2+((v-v0)/alpha_v)**2
                u = (u-u0)*(1+k1*rd2+k2*(rd2)**2)+u0
                v = (v-v0)*(1+k1*rd2+k2*(rd2)**2)+v0
            epsilon = (u-I[i][0])**2 + (v-I[i][1])**2
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
    
    def iterative_refienment(self, radial_distortion:bool):
        def fr(x:np.ndarray, *args, **kwargs):
            real_coords = self.real_coords[:,np.r_[0,1,3]]
            K, dist_par, R, t = self._unpack_parameters(x, True)
            k1 = dist_par[0]
            k2 = dist_par[1]
            residuals = np.empty((0,1))
            for i in range(self._no_of_images):
                H = K@(np.hstack((R[i][:,0].reshape(-1,1), R[i][:,1].reshape(-1,1), t[i].reshape(-1,1))))
                proj_coords = np.transpose(H@real_coords.T)
                skewness = np.arctan(K[0,0]/K[0,1])
                alpha_u, alpha_v = K[0,0], np.abs(K[1,1]*np.sin(skewness))
                u0, v0 = K[0,2], K[1,2]
                u = proj_coords[:,0]/proj_coords[:,2]
                v = proj_coords[:,1]/proj_coords[:,2]
                rd2 = ((u-u0)/alpha_u)**2+((v-v0)/alpha_v)**2
                uh = (u-u0)*(1+k1*rd2+k2*(rd2)**2)+u0
                vh = (v-v0)*(1+k1*rd2+k2*(rd2)**2)+v0
                distorted = np.hstack((uh.reshape(-1,1), vh.reshape(-1,1)))
                chunk_residual = self.pix_coords[i]-distorted
                chunk_residual = np.linalg.norm(chunk_residual, axis=1)
                residuals = np.vstack((residuals, chunk_residual.reshape(-1,1)))
            return residuals.flatten()
        def fnr(x:np.ndarray, *args, **kwargs):
            real_coords = self.real_coords[:,np.r_[0,1,3]]
            K, R, t = self._unpack_parameters(x, False)
            residuals = np.empty((0,1))
            for i in range(self._no_of_images):
                H = K@np.hstack((R[i][:,0].reshape(-1,1), R[i][:,1].reshape(-1,1), t[i].reshape(-1,1)))
                proj_coords = np.transpose(H@real_coords.T)
                u = proj_coords[:,0]/proj_coords[:,2]
                v = proj_coords[:,1]/proj_coords[:,2]
                distorted = np.hstack((u.reshape(-1,1), v.reshape(-1,1)))
                chunk_residual = self.pix_coords[i]-distorted
                chunk_residual = np.linalg.norm(chunk_residual, axis=1)
                residuals = np.vstack((residuals, chunk_residual.reshape(-1,1)))
            return residuals.flatten()
        x0 = self._pack_parameters(radial_distortion)
        if radial_distortion is True:
            result = least_squares(fr, x0, method='lm')
            K, dist_par, R, t = self._unpack_parameters(result.x, True)
            self.distortion_parameters = dist_par
        else:
            result = least_squares(fnr, x0, method='lm')
            K, R, t = self._unpack_parameters(result.x, False)
        
        projections = []
        for i in range(self._no_of_images):
            U, _ , V_t = np.linalg.svd(R[i])
            R[i] = U@V_t
            P = K@np.hstack((R[i], t[i].reshape(-1,1)))
            projections.append(P)
            
        self.K = K
        self.rotations = R
        self.translations = t
        self.projection_matrices = projections
        return
 
    def fit(self, images:list[np.ndarray], radial_distortion=False, iterative=False):
        self.reset()
        self._no_of_images = len(images)
        for image in images:
            H, real_coords, pix_coords = self.estimate_homography(image)
            if self.real_coords is None:
                self.real_coords = real_coords
            self.pix_coords.append(pix_coords)
            self.homographies.append(H)
        V = self.calculate_V()
        self.estimate_K(V)        
        for i,H in enumerate(self.homographies):
            P, R, t = self.estimate_projection_matrix(H, self.K)
            if P[2,3] < 0:
                self.homographies[i] = -H
                P, R, t = self.estimate_projection_matrix(-H, self.K)
            self.projection_matrices.append(P)
            self.rotations.append(R)
            self.translations.append(t)
        if radial_distortion is True:
            self.estimate_distortion()
        if iterative is True:
            self.iterative_refienment(radial_distortion)
        for i,P in enumerate(self.projection_matrices):
            error = self.reprojection_error(P, self.real_coords, self.pix_coords[i], radial_distortion)
            self.reprojection_errors.append(error)

    def compensate_radial_distortion(self, image:np.ndarray) -> np.ndarray:
        compensated = np.empty_like(image)
        K = self.K
        skewness = np.arctan(K[0,0]/K[0,1])
        alpha_u, alpha_v = K[0,0], np.abs(K[1,1]*np.sin(skewness))
        u0, v0 = K[0,2], K[1,2]
        k1 = self.distortion_parameters[0]
        k2 = self.distortion_parameters[1]
        map_x = np.empty_like(image, dtype=np.float32)
        map_y = np.empty_like(image, dtype=np.float32)
        for u in range(compensated.shape[1]):
            for v in range(compensated.shape[0]):
                rd2 = ((u-u0)/alpha_u)**2+((v-v0)/alpha_v)**2
                map_x[v,u] = (u-u0)*(1+k1*rd2+k2*(rd2)**2)+u0
                map_y[v,u] = (v-v0)*(1+k1*rd2+k2*(rd2)**2)+v0
        compensated = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC)
        return compensated

    def _vij_function (self, H, i, j):
        v=np.zeros(6)
        v[0]=H[0][i]*H[0][j]
        v[1]=H[0][i]*H[1][j]+H[1][i]*H[0][j]
        v[2]=H[1][i]*H[1][j]
        v[3]=H[2][i]*H[0][j]+H[0][i]*H[2][j]
        v[4]=H[2][i]*H[1][j]+H[1][i]*H[2][j]
        v[5]=H[2][i]*H[2][j]
        return v
    
    def _pack_parameters(self, radial:bool):
        flattened_K = self.K.flatten()
        if radial is True:
            x = np.append(flattened_K[np.r_[0:0+3,4,5]], self.distortion_parameters)
        else:
            x = flattened_K[np.r_[0:0+3,4,5]]
        for i in range(self._no_of_images):
            r = Rotation.from_matrix(self.rotations[i])
            r = r.as_mrp()
            x = np.append(x,r)
            t = self.translations[i]
            x = np.append(x,t)
        return x
    
    def _unpack_parameters(self, x:np.ndarray, radial:bool):
        flattened_K = x[0:5]
        K = np.eye(3,3)
        K[0,:] = flattened_K[0:0+3]
        K[1,1:] = flattened_K[3:]
        R = []
        t = []
        if radial is True:
            distortion_params = x[5:7]
            for i in range(7, x.shape[0], 6):
                rot = x[i:i+3]
                rot = Rotation.from_mrp(rot)
                R.append(rot.as_matrix())
                t.append(x[i+3:i+6])
            return K, distortion_params, R, t
        else:
            for i in range(5, x.shape[0], 6):
                rot = x[i:i+3]
                rot = Rotation.from_mrp(rot)
                R.append(rot.as_matrix())
                t.append(x[i+3:i+6])
            return K, R, t