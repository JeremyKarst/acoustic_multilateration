# Copyright (C) 2024 Jeremy Karst - publishing.recolor515@passmail.net
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# 

import numpy as np
import scipy as sp

class RobustCovariance:
    def __init__(self, support_fraction=0.75, max_iter=100):
        self.support_fraction = support_fraction
        self.max_iter = max_iter
        self.location_ = None
        self.covariance_ = None
        self.support_ = None
    
    def _select_initial_subset(self, X):
        """Select initial subset using spatial median"""
        n_samples, n_features = X.shape
        spatial_median = np.median(X, axis=0)
        
        # Calculate distances to spatial median
        distances = np.sum((X - spatial_median)**2, axis=1)
        h = int(n_samples * self.support_fraction)
        
        # Select h points closest to spatial median
        initial_subset = np.argpartition(distances, h)[:h]
        return initial_subset
    
    def _estimate_location_covariance(self, X, subset, sample_weight=None):
        """Estimate location and covariance for given subset"""
        if sample_weight is None:
            sample_weight = np.ones(len(subset))
        else:
            sample_weight = np.asarray(sample_weight[subset])
            if len(sample_weight) != len(subset):
                raise ValueError("sample_weight must have the same length as subset")
        
        location = np.average(X[subset], axis=0, weights=sample_weight)
        centered_X = X[subset] - location
        covariance = np.cov(centered_X, rowvar=False, aweights=sample_weight)
        return location, covariance
    def _mahalanobis_distances(self, X, location, covariance):
        """Calculate Mahalanobis distances"""
        try:
            inv_cov = np.linalg.inv(covariance)
            centered_X = X - location
            dist = np.sum(centered_X @ inv_cov * centered_X, axis=1)
            return np.sqrt(dist)
        except np.linalg.LinAlgError:
            return np.full(len(X), np.inf)
    
    def fit(self, X, sample_weight=None):
        n_samples, n_features = X.shape
        h = int(n_samples * self.support_fraction)
        
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.asarray(sample_weight)
            if len(sample_weight) != n_samples:
                raise ValueError("sample_weight must have the same length as X")
            sample_weight[sample_weight == 0] = 1e-10
        
        # Get initial subset
        best_subset = self._select_initial_subset(X)
        best_location, best_covariance = self._estimate_location_covariance(X, best_subset, sample_weight)
        best_det = np.linalg.det(best_covariance)
        
        for _ in range(self.max_iter):
            # Calculate distances
            distances = self._mahalanobis_distances(X, best_location, best_covariance)
            
            # Select h points with smallest weighted distances
            weighted_distances = distances / sample_weight
            new_subset = np.argpartition(weighted_distances, h)[:h]
            
            # Calculate new estimates
            new_location, new_covariance = self._estimate_location_covariance(X, new_subset, sample_weight)
            new_det = np.linalg.det(new_covariance)
            
            # Check for convergence
            if np.isclose(new_det, best_det, rtol=1e-7):
                break
                
            if new_det < best_det:
                best_subset = new_subset
                best_location = new_location
                best_covariance = new_covariance
                best_det = new_det
        
        self.location_ = best_location
        self.covariance_ = best_covariance
        self.support_ = best_subset
        return self
    
def robust_covariance_filter(tdoa_data, confidences, window_size=21, support_fraction=0.75, expected_outlier_rate=0.3):
    """
    Robust covariance filtering for TDOA data
    
    Parameters:
    tdoa_data: array of shape (n_timestamps, n_pairs)
    confidences: array of shape (n_timestamps,) containing confidence values for each tdoa_data
    window_size: int, size of sliding window
    support_fraction: float, fraction of data to consider inliers
    
    Returns:
    Filtered TDOA data
    """
    assert window_size % 2 == 1, "Window size must be odd"
    n_times, n_pairs = tdoa_data.shape
    filtered_data = tdoa_data.copy()
    
    # Calculate chi-square threshold for outlier detection
    chi2_thresh = sp.stats.chi2.ppf(1-expected_outlier_rate, df=n_pairs)

    
    # Process each window
    half_window = window_size // 2

    for t in range(half_window, n_times - half_window):
        # Extract current window
        window = tdoa_data[t-half_window:t+half_window+1, :]
        window_confidences = confidences[t-half_window:t+half_window+1]
        if window_confidences.sum() == 0: # Handle cases where all confidences are zero within the window range.
            window_confidences = np.ones_like(window_confidences)
        
        # Fit robust covariance estimator, weighted by confidences
        robust_cov = RobustCovariance(support_fraction=support_fraction)
        
        try:
            robust_cov.fit(window, sample_weight=window_confidences)
            
            # Get current sample and confidence
            curr_sample = tdoa_data[t, :]
            curr_confidence = confidences[t]
            
            # Calculate Mahalanobis distance
            diff = curr_sample - robust_cov.location_
            inv_cov = np.linalg.inv(robust_cov.covariance_)
            mahal_dist = np.sqrt(diff @ inv_cov @ diff.T)
            
            # Replace outliers with robust estimate
            if mahal_dist > chi2_thresh:
                filtered_data[t, :] = robust_cov.location_
                
        except Exception as e:
            # This will happen if the covariance matrix is singular
            continue
    
    return filtered_data

def analyze_outliers(tdoa_data, window_size=20, support_fraction=0.75):
    """
    Analyze outliers in TDOA data and return diagnostics
    """
    n_times, n_pairs = tdoa_data.shape
    outlier_mask = np.zeros(n_times, dtype=bool)
    mahal_distances = np.zeros(n_times)
    
    chi2_thresh = n_pairs * (1 - 2/(9*n_pairs) + np.sqrt(2/(9*n_pairs)) * 2.326347)**3
    half_window = window_size // 2
    
    for t in range(half_window, n_times - half_window):
        window = tdoa_data[t-half_window:t+half_window, :].T
        robust_cov = RobustCovariance(support_fraction=support_fraction)
        
        try:
            robust_cov.fit(window)
            curr_sample = tdoa_data[t, :]
            diff = curr_sample - robust_cov.location_
            inv_cov = np.linalg.inv(robust_cov.covariance_)
            mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
            
            mahal_distances[t] = mahal_dist
            outlier_mask[t] = mahal_dist > chi2_thresh
            
        except:
            continue
            
    return outlier_mask, mahal_distances

class KalmanPositionFilter:
    def __init__(self, max_acceleration, dt, base_radial_noise, base_angular_noise_rad, distance_factor=0.009):
        """
        Initialize Kalman filter for 2D position tracking with polar coordinate noise model
        
        Args:
            max_acceleration (float): Maximum expected acceleration in units/second^2
            dt (float): Time step between measurements in seconds
            base_radial_noise (float): Base standard deviation of radial measurement noise
            base_angular_noise_rad (float): Base standard deviation of angular noise in radians
            distance_factor (float): How much radial noise increases with distance
        """
        # State vector: [x, y, vx, vy]
        self.state = np.zeros(4)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        q = (max_acceleration * dt**2 / 2)**2
        self.Q = np.array([
            [q/4, 0, q/(2*dt), 0],
            [0, q/4, 0, q/(2*dt)],
            [q/(2*dt), 0, q, 0],
            [0, q/(2*dt), 0, q]
        ])
        
        # Store noise parameters
        self.base_radial_noise = base_radial_noise
        self.base_angular_noise = base_angular_noise_rad
        self.distance_factor = distance_factor
        
        # Initial state covariance
        self.P = np.eye(4) * 1000  # Large initial uncertainty
        
        self.initialized = False
    
    def _get_measurement_noise(self, measurement):
        """
        Calculate measurement noise covariance in Cartesian coordinates
        based on polar coordinate noise model
        """
        x, y = measurement
        distance = np.linalg.norm(measurement)
        
        if distance < 1e-6:  # Avoid division by zero
            return np.eye(2) * self.base_radial_noise**2
        
        # Calculate polar angles
        theta = np.arctan2(y, x)
        
        # Get radial noise (increases with distance)
        radial_std = self.base_radial_noise * (1 + self.distance_factor * distance)
        
        # Angular noise in distance units (converts angular uncertainty to position uncertainty)
        # This increases with distance since same angle covers more distance at range
        tangential_std = distance * self.base_angular_noise
        
        # Convert polar uncertainties to Cartesian covariance matrix
        cos_theta = x / distance
        sin_theta = y / distance
        
        # Rotation matrix from polar to Cartesian uncertainties
        R = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        # Polar coordinate uncertainties
        D = np.array([
            [radial_std**2, 0],
            [0, tangential_std**2]
        ])
        
        # Convert to Cartesian coordinate uncertainties
        return R @ D @ R.T
    
    def update(self, measurement):
        """
        Update state estimate with new position measurement
        
        Args:
            measurement: numpy array [x, y] with new position measurement
            
        Returns:
            numpy array [x, y] with filtered position estimate
        """
        measurement = np.asarray(measurement)
        
        if not self.initialized:
            self.state[:2] = measurement
            self.initialized = True
            return measurement
        
        # Predict
        state_pred = self.F @ self.state
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # Update with position-dependent noise
        R = self._get_measurement_noise(measurement)
        y = measurement - (self.H @ state_pred)
        S = self.H @ P_pred @ self.H.T + R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        self.state = state_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred
        
        return self.state[:2]
    
    def smooth_trajectory(self, measurements):
        """
        Apply Kalman filtering to entire trajectory
        
        Args:
            measurements: numpy array of shape (n, 2) containing position measurements
            
        Returns:
            numpy array of shape (n, 2) containing smoothed positions
        """
        smoothed = np.zeros_like(measurements)
        for i in range(len(measurements)):
            smoothed[i] = self.update(measurements[i])
        return smoothed

def filter_positions(positions, dt, max_acceleration, base_radial_noise, base_angular_noise, distance_factor):
    """
    Convenience function to filter a sequence of positions
    
    Args:
        positions: numpy array of shape (n, 2) containing [x, y] measurements
        dt: Time step between measurements in seconds
        max_acceleration: Maximum expected acceleration in units/second^2
        base_radial_noise: Standard deviation of radial measurement noise at origin
        base_angular_noise: Standard deviation of angular noise (in radians)
        distance_factor: How much radial noise increases with distance
        
    Returns:
        numpy array of shape (n, 2) containing filtered positions
    """
    # Convert angular noise to radians if needed
    base_angular_noise_rad = base_angular_noise
    
    kf = KalmanPositionFilter(
        max_acceleration, 
        dt, 
        base_radial_noise,
        base_angular_noise_rad,
        distance_factor
    )
    return kf.smooth_trajectory(positions)