"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        
        pad_x_left_image = win_size // 2
        pad_x_right_image = dsp_range + win_size // 2
        pad_y = win_size // 2
        left_image_padded = np.pad(left_image, ((pad_y, pad_y), (pad_x_left_image, pad_x_left_image), (0, 0)))
        right_image_padded = np.pad(right_image, ((pad_y, pad_y), (pad_x_right_image, pad_x_right_image), (0, 0)))
        
        num_of_padded_rows, num_of_padded_cols = left_image_padded.shape[0], left_image_padded.shape[1]
        sum_map = np.zeros((num_of_padded_rows, num_of_padded_cols, 3, dsp_range*2+1))
        for idx in range(len(disparity_values)):
            sum_map[..., idx] = right_image_padded[:, idx:idx + num_of_padded_cols, :] - left_image_padded
        sum_map = sum_map**2
        
        kernel = np.ones((win_size, win_size))
        for i in range(len(disparity_values)):
            for channel_idx in range(3):
                ssdd_tensor[:, :, i] += convolve2d(sum_map[:,:,channel_idx,i], kernel, mode='valid')
        
        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        # you can erase the label_no_smooth initialization.
        label_no_smooth = np.argmin(ssdd_tensor,axis=-1)
        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))
        for col in range(num_of_cols):
            if col == 0:
                l_slice[:, 0] = c_slice[:, 0]
                continue
            
            a = l_slice[:, col-1]
            
            b_temp = np.full((2, num_labels), np.inf)
            b_temp[0, 1:] = l_slice[:-1, col-1]
            b_temp[0, :-1] = l_slice[1:, col-1]
            b = p1 + b_temp.min(axis=0)
            
            c_temp = np.full((num_labels, num_labels), np.inf)
            for idx, k in enumerate(range(-(num_labels//2), num_labels//2 + 1)):
                if k < -1:
                    c_temp[idx, -k:] = l_slice[:k, col-1]
                elif k > 1:
                    c_temp[idx, :-k] = l_slice[k:, col-1]
            c = p2 + c_temp.min(axis=0)
            
            l_slice[:, col] = c_slice[:, col] + np.minimum(np.minimum(a, b), c) - np.min(c_slice[:, col - 1])
        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        H, W, D = l.shape
        for row in range(H):
            l[row] = self.dp_grade_slice(ssdd_tensor[row].transpose(), p1, p2).transpose()
        return self.naive_labeling(l)
    
    def calc_score_row(self, ssdd_tensor: np.ndarray, p1: float, p2: float, 
                        horizontal_flip: bool, vertical_flip: bool) -> np.ndarray:
        l = np.zeros_like(ssdd_tensor)
        H, W, D = l.shape
        for row in range(H):
            is_flip = -1 if horizontal_flip else 1
            l[row] = self.dp_grade_slice(ssdd_tensor[row, ::is_flip].transpose(), p1, p2).transpose()
        l = l[:, ::is_flip]
        return l

    def calc_score_diag(self, ssdd_tensor: np.ndarray, p1: float, p2: float, 
                         horizontal_flip: bool, vertical_flip: bool) -> np.ndarray:
        l = np.zeros_like(ssdd_tensor)
        H, W, D = l.shape
        grid = np.arange(H*W).reshape(H, W)
        grid = np.flipud(grid) if vertical_flip else grid
        grid = np.fliplr(grid) if horizontal_flip else grid
        
        for diag_idx in range(-H+1, W):
            idxs = np.unravel_index(np.diag(grid, diag_idx), (H, W))
            l[idxs] = self.dp_grade_slice(ssdd_tensor[idxs].transpose(), p1, p2).transpose()
        return l
    
    def calc_score_col(self, ssdd_tensor: np.ndarray, p1: float, p2: float, 
                        horizontal_flip: bool, vertical_flip: bool) -> np.ndarray:
        l = np.zeros_like(ssdd_tensor)
        H, W, D = l.shape
        is_flip = -1 if vertical_flip else 1
        for col in range(W):
            l[:, col] = self.dp_grade_slice(ssdd_tensor[::is_flip, col].transpose(), p1, p2).transpose()
        l = l[::is_flip]
        return l
    
    def calc_score_per_direction(self, ssdd_tensor: np.ndarray, p1: float, p2: float, direction: int):
        horizonal_flip = True if 4 <= direction <= 6 else False
        vertical_flip = True if 6 <= direction <= 8 else False
        if direction in [1,5]:
            func = self.calc_score_row
        elif direction in [2,4,6,8]:
            func = self.calc_score_diag
        else:
            func = self.calc_score_col
        return func(ssdd_tensor, p1 ,p2, horizonal_flip, vertical_flip)

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        for direction in range(1, num_of_directions + 1):
            direction_to_slice[direction] = self.naive_labeling(self.calc_score_per_direction(ssdd_tensor, p1, p2, direction))
        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        l_direction = None
        for direction in range(1, num_of_directions + 1):
            curr_l = self.calc_score_per_direction(ssdd_tensor, p1, p2, direction)
            l_direction = curr_l[None] if l_direction is None else np.vstack((l_direction, curr_l[None]))
        l = l_direction.mean(axis=0)
        return self.naive_labeling(l)

