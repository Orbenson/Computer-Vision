"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        src_p_count = match_p_src.shape[1]
        dst_p_count = match_p_dst.shape[1]
        assert (src_p_count == dst_p_count and src_p_count >= 4)
        
        A = list()
        x, y = match_p_src[0], match_p_src[1]
        u, v = match_p_dst[0], match_p_dst[1]
        for i in range(src_p_count):
            x_i, y_i = x[i], y[i]
            u_i, v_i = u[i], v[i]
            A.append([x_i, y_i, 1, 0, 0, 0, -u_i * x_i, -u_i * y_i, -u_i])
            A.append([0, 0, 0, x_i, y_i, 1, -v_i * x_i, -v_i * y_i, -v_i])
        A = np.array(A)

        _, _, Vh = svd(A)
        min_eigval = Vh[-1] # Last row of Vh is the smallest eigenvector
        h = min_eigval / np.linalg.norm(min_eigval) # Vector normaliation
        
        H = np.reshape(h, (3,3))
        return H

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        x, y, c = [], [], []
        H_s, W_s, _ = src_image.shape
        H_d, W_d, _ = dst_image_shape
        
        for i in range(H_s):
            for j in range(W_s):
                pix_rgb = src_image[i, j, :] # Pixel channels value
                pix_coords = np.array([j, i, 1]) # Pixel homogenous coordinates
                
                j_prime, i_prime, scale = homography.dot(pix_coords)
                j_prime = j_prime / scale
                i_prime = i_prime / scale
                
                x.append(j_prime)
                y.append(i_prime)
                c.append(pix_rgb)
        
        x = np.array(x)
        y = np.array(y)
        c = np.stack(c, axis=0)
        
        outlier_radius = 1000
        is_inlier_x = np.abs(x - x.mean()) < outlier_radius
        is_inlier_y = np.abs(y - y.mean()) < outlier_radius
        is_inlier = np.bitwise_and(is_inlier_x, is_inlier_y)
        
        x = x[is_inlier]
        y = y[is_inlier]
        c = c[is_inlier]
        
        x = (x - x.min()).astype(int)
        y = (y - y.min()).astype(int)
        
        warp_img = np.zeros(dst_image_shape, dtype=np.uint8)
        in_frame_x = np.bitwise_and(0 <= x, x < W_d)
        in_frame_y = np.bitwise_and(0 <= y, y < H_d)
        in_frame = np.bitwise_and(in_frame_x, in_frame_y)
        warp_img[y[in_frame], x[in_frame], :] = c[in_frame]
        
        return warp_img

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        H_s, W_s, _ = src_image.shape
        
        x, y = np.meshgrid(np.arange(W_s), np.arange(H_s))
        x, y = x.reshape(-1), y.reshape(-1)
        homogenoues_coords = np.stack((x, y, np.ones_like(y)))
        homogenoues_coords_prime = homography.dot(homogenoues_coords)
        xy = (homogenoues_coords_prime / homogenoues_coords_prime[-1, :])[:-1, :]
        
        outlier_radius = 1000
        is_inlier = np.abs(xy - xy.mean(axis=1).reshape(-1,1)) < outlier_radius
        is_inlier = np.bitwise_and(is_inlier[0, :], is_inlier[1, :])
        
        xy = xy[:, is_inlier]
        c = src_image.reshape(-1, 3)
        c = c[is_inlier]
        
        min_xy = xy.min(axis=1).reshape(-1,1)
        H_d, W_d, _ = dst_image_shape

        xy = (xy - min_xy).astype(int)
        in_frame_x = np.bitwise_and(0 <= xy[0, :], xy[0, :] < W_d)
        in_frame_y = np.bitwise_and(0 <= xy[1, :], xy[1, :] < H_d)
        in_frame = np.bitwise_and(in_frame_x, in_frame_y)
        
        xy = xy[:, in_frame]
        c = c[in_frame]
        
        warp_img = np.zeros(dst_image_shape, dtype=np.uint8)
        warp_img[xy[1,:], xy[0,:], :] = c
        return warp_img

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        N = match_p_src.shape[1]
        
        homogenoues_src_p = np.vstack((match_p_src, np.ones(N)))
        warp_src_p = homography.dot(homogenoues_src_p)
        warp_src_p = (warp_src_p / warp_src_p[-1, :])[:-1, :]
        
        dist = np.linalg.norm(warp_src_p - match_p_dst, ord=2, axis=0)
        valid = np.sum(dist <= max_err)
        fit_percentage = valid / N
        dist_mse = dist.sum() / N

        return fit_percentage, dist_mse

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        N = match_p_src.shape[1]
        
        homogenoues_src_p = np.vstack((match_p_src, np.ones(N)))
        warp_src_p = homography.dot(homogenoues_src_p)
        warp_src_p = (warp_src_p / warp_src_p[-1, :])[:-1, :]
        
        dist = np.linalg.norm(warp_src_p - match_p_dst, ord=2, axis=0)
        is_valid = dist <= max_err
        
        mp_src_meets_model = match_p_src[:, is_valid]
        mp_dst_meets_model = match_p_dst[:, is_valid]
        
        return mp_src_meets_model, mp_dst_meets_model

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        w = inliers_percent
        t = max_err
        p = 0.99 # parameter determining the probability of the algorithm to succeed
        d = 0.5 # the minimal probability of points which meets with the model
        n = 4 # number of points sufficient to compute the model
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1 # number of RANSAC iterations (+1 to avoid the case where w=1)
        
        N = match_p_src.shape[1]
        best_mse = np.inf
        for i in range(k):
            idxs = sample(range(N), k=n)
            mp_src_sampled = match_p_src[:, idxs]
            mp_dst_sampled = match_p_dst[:, idxs]
            
            H = self.compute_homography_naive(mp_src_sampled, mp_dst_sampled) # homography based on random idxs
            mp_src_meets_model, mp_dst_meets_model = self.meet_the_model_points(H, match_p_src, match_p_dst, t)
            if (mp_src_meets_model.shape[1] / N > d or i == 0):
                H = self.compute_homography_naive(mp_src_meets_model, mp_dst_meets_model)
                _, dist_mse = self.test_homography(H, mp_src_meets_model, mp_dst_meets_model, t)
                if dist_mse > best_mse:
                    continue
                best_mse = dist_mse
                best_H = H
        
        return best_H

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """
        H_d, W_d, _ = dst_image_shape
        
        x_d, y_d = np.meshgrid(np.arange(W_d), np.arange(H_d))
        x_d, y_d = x_d.reshape(-1), y_d.reshape(-1)
        homogenoues_coords = np.stack((x_d, y_d, np.ones_like(y_d)))
        homogenoues_coords_prime = backward_projective_homography.dot(homogenoues_coords)
        coords_prime = homogenoues_coords_prime[:-1, :] / homogenoues_coords_prime[-1, :]
        coords_prime = coords_prime.transpose()
        
        H_s, W_s, _ = src_image.shape
        x_s, y_s = np.meshgrid(np.arange(W_s), np.arange(H_s))
        x_s, y_s = x_s.reshape(-1), y_s.reshape(-1)
        xy = np.stack((x_s, y_s)).transpose()
        
        r_s = src_image[:,:,0].reshape(-1)
        g_s = src_image[:,:,1].reshape(-1)
        b_s = src_image[:,:,2].reshape(-1)
        
        backward_warp_r = griddata(xy, r_s, coords_prime, method="cubic", fill_value=0)
        backward_warp_g = griddata(xy, g_s, coords_prime, method="cubic", fill_value=0)
        backward_warp_b = griddata(xy, b_s, coords_prime, method="cubic", fill_value=0)
        
        backward_warp = np.stack((backward_warp_r, backward_warp_g, backward_warp_b))
        backward_warp = backward_warp.transpose().reshape(H_d, W_d, 3)
        return backward_warp

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([0, 0, 1])
        src_edges['upper right corner'] = np.array([src_cols_num - 1, 0, 1])
        src_edges['lower left corner'] = np.array([0, src_rows_num - 1, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num - 1, src_rows_num - 1, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 0:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num - 1:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - (dst_cols_num - 1)])
            if corner_location[0] < 0:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num - 1:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - (dst_rows_num - 1)])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        T = np.eye(3)
        T[0, 2] = -pad_left
        T[1, 2] = -pad_up
        
        final_homography = backward_homography.dot(T)
        final_homography = final_homography / final_homography[2,2]
        return final_homography

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        H = self.compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)
        H_p, W_p, padding = self.find_panorama_shape(src_image, dst_image, H)
        
        inv_H = np.linalg.inv(H)
        inv_H = self.add_translation_to_backward_homography(inv_H, padding.pad_left, padding.pad_up)
        warp_img = self.compute_backward_mapping(inv_H, src_image ,(H_p, W_p, 3))
        
        img_panorama = np.zeros((H_p, W_p, 3))
        
        H_d, W_d, _ = dst_image.shape
        dest_loc_h = H_p - H_d - padding.pad_down
        dest_loc_w = W_p - W_d - padding.pad_right
        img_panorama[dest_loc_h:dest_loc_h + H_d, dest_loc_w:dest_loc_w + W_d, :] = dst_image
        
        H_w, W_w, _ = warp_img.shape
        panorama_black_idxs = np.where(img_panorama[:H_w, :W_w, :] == (0,0,0))
        img_panorama[panorama_black_idxs] = warp_img[panorama_black_idxs]
        
        
        return np.clip(img_panorama, 0, 255).astype(np.uint8)
