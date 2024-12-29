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
import matplotlib.pyplot as plt
import scipy as sp

import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

class Multilateration:
    """
    Multilateration class for estimating the position of a sound source using TDOA measurements.
    """
    def __init__(self, mic_positions, sample_rate, c, max_distance = None, method = "correlation", corr_smoothing = None, low_freq_cutoff = 50, high_freq_cutoff = 1000, confidence_metric = 'min'):
        """ 
        Initialize the Multilateration class.

        Args:
            mic_positions (list of np.ndarray): List of microphone positions [x,y]
            sample_rate (int): Sample rate of the signals in Hz
            c (float): Speed of sound in m/s
            max_distance (float, optional): Maximum distance from the centroid of the microphones to the source, prevents invalid results very far from sensors (default is None)
            method (str, optional): Method to use for TDOA estimation ("correlation" or "gcc") (default is "correlation")
            corr_smoothing (int, optional): Smoothing factor for the correlation function (default is None)
            low_freq_cutoff (float, optional): Low frequency cutoff for the GCC-PHAT method (default is 50 Hz)
            high_freq_cutoff (float, optional): High frequency cutoff for the GCC-PHAT method (default is 1000 Hz)
            confidence_metric (str, optional): Confidence metric to use for TDOA estimation ("min", "mean", "median") (default is "min")
        """
        self.mic_positions = np.array(mic_positions)
        self.sample_rate = sample_rate
        self.speed_of_sound = c
        self.method = method
        self.corr_smoothing = corr_smoothing
        self.low_freq_cutoff = low_freq_cutoff
        self.high_freq_cutoff = high_freq_cutoff
        self.confidence_metric = confidence_metric

        self.confidence_percentile = 95 # This is where we assume the noise floor of the correlation result is

        assert self.confidence_metric in ["min", "mean", "median"], "Invalid confidence metric, valid options are 'min', 'mean', or 'median'"

        assert (corr_smoothing is None) or (corr_smoothing % 2 == 1), "Smoothing must be None or an odd number to prevent phase shift"

        if self.method not in ["corr", "gcc"]:
            raise ValueError(f"Invalid method: {self.method}")
        
        self.max_distance = max_distance # Maximum distance from the centroid of the microphones to the source, prevents invalid results very far from sensors
        
        microphone_local_x = mic_positions[:,0]
        microphone_local_y = mic_positions[:,1]
        distances = np.sqrt((microphone_local_x[:,None] - microphone_local_x[None,:])**2 + (microphone_local_y[:,None] - microphone_local_y[None,:])**2)
        self.max_tau = np.max(distances) / self.speed_of_sound * 1.1 # A little bit of fudge factor to account for multipath effects, and non-perfect estimate of speed of sound.

    def gcc_phat(self, a, b, weight=10.0, return_correlation = False):
        """
        Generalized Cross Correlation with Phase Transform (GCC-PHAT)
        
        Args:
            a (np.ndarray): Signal from the first microphone (or sensor)
            b (np.ndarray): Signal from the second microphone (or sensor)
            weight (float, optional): Weight to apply to the frequency band (default is 10.0 / no weighting)
            return_correlation (bool, optional): Whether to return the correlation array (for debugging) (default is False)
        
        Returns:
            tdoa (float): Time difference of arrival between the signals
            confidence (float, optional): Confidence in the TDOA estimate
        """

        # Ensure the signals are the same length by zero-padding the shorter one
        n = a.shape[0] + b.shape[0]
        window = np.hanning(len(a))
        SIG1 = np.fft.fft(a * window, n=n)
        SIG2 = np.fft.fft(b * window, n=n)
        
        # Frequency vector (for identifying the frequencies in the FFT)
        freqs = np.fft.fftfreq(n, 1 / self.sample_rate)
        
        # Band-pass weighting: apply more emphasis to freq1-freq2
        # Create a weight mask with higher values in the specified band
        weight_mask = np.ones(n)
        if self.low_freq_cutoff is not None and self.high_freq_cutoff is not None and weight != 1.0:
            # Apply the weight to frequencies between freq1 and freq2
            mask = (np.abs(freqs) >= self.low_freq_cutoff) & (np.abs(freqs) <= self.high_freq_cutoff)
            weight_mask[mask] = weight
        
        # Cross-power spectrum
        R = SIG1 * np.conj(SIG2)
        
        # Apply Phase Transform (PHAT)
        R = R * weight_mask / np.abs(R)
        
        # Cross-correlation by inverse FFT and take the real part
        cc = np.real(np.fft.ifft(R))
        
        # Shift result for proper correlation around zero lag
        max_shift = int(n // 2)
        cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1])) 
        if self.corr_smoothing:
            cc = np.convolve(cc, np.ones(self.corr_smoothing)/self.corr_smoothing, mode='same')
        
        # Find the valid range based on max_tau
        max_tau_samples = int(self.max_tau * self.sample_rate)
        valid_range = np.arange(max_shift - max_tau_samples, max_shift + max_tau_samples + 1)
        
        # Find the index of the max cross-correlation value within the valid range
        max_index = valid_range[np.argmax(cc[valid_range])]
        delay_index = max_index - max_shift

        tau = delay_index / self.sample_rate

        # # Calculate confidence metric
        peak_value = cc[max_index]

        # Try to calculate a confidence metric based on the differences between peak values for the highest and second highest peaks
        # Make sure that the second peak is at least one zero-crossing away from the highest peak
        # Find the zero crossings in the correlation function
        zero_crossings = np.where(np.diff(np.sign(cc[valid_range])))[0]
        
        index_of_max_within_valid_range = max_index - valid_range[0]
        # Find the index of the second highest peak that is at least one zero crossing away from the main peak
        sorted_indices = np.argsort(cc[valid_range])[::-1]
        for idx in sorted_indices:
            if idx == index_of_max_within_valid_range:
                continue
            zero_crossings_between = np.logical_and(zero_crossings > min(idx, index_of_max_within_valid_range), zero_crossings < max(idx, index_of_max_within_valid_range))
            if np.any(zero_crossings_between):
                second_peak_index = idx
                break

        # # Plot the valid range of the correlation function, as well as the highest and second highest peaks
        # plt.figure()
        # plt.plot(cc[valid_range])
        # plt.scatter(index_of_max_within_valid_range, cc[valid_range][index_of_max_within_valid_range], color='red', label='Highest Peak')
        # plt.scatter(second_peak_index, cc[valid_range][second_peak_index], color='blue', label='Second Highest Peak')
        # plt.legend()
        # plt.show()

        if not second_peak_index: # If we did not find a second peak for our confidence metric, set confidence to 0
            confidence = 0.0
        elif np.abs(tau) > self.max_tau: # If our estimate is outside of the valid range, set confidence to 0
            confidence = 0.0
        else:
            # Calculate the second confidence metric
            second_peak_value = cc[valid_range][second_peak_index]
            noise_floor_percentile = np.percentile(cc[valid_range], self.confidence_percentile) # Percentile works better than standard deviation for this metric due to the non-Gaussian nature of the correlation function
            peak_above_noise_floor = peak_value - noise_floor_percentile
            second_peak_above_noise_floor = max(0.0, second_peak_value - noise_floor_percentile)
            confidence = 1 - (second_peak_above_noise_floor / peak_above_noise_floor)

        if return_correlation:
            return cc, tau, confidence
        else:
            return tau, confidence
    
    def correlation_tdoa(self, a, b, return_correlation = False):
        """
        Direct Cross-Correlation (TDOA)
        
        Args:
            a (np.ndarray): Signal from the first microphone (or sensor)
            b (np.ndarray): Signal from the second microphone (or sensor)
            return_correlation (bool, optional): Whether to return the correlation array (for debugging) (default is False)
        
        Returns:
            tdoa (float): Time difference of arrival between the signals
            confidence (float [0,1], optional): Confidence in the TDOA estimate
        """
        # Calculate correlation
        cc = sp.signal.correlate(a, b, mode='full')

        if self.corr_smoothing:
            cc = np.convolve(cc, np.ones(self.corr_smoothing)/self.corr_smoothing, mode='same')

        lags = np.arange(-(len(a) - 1), len(b))
        
        # If max_tau is provided, limit the search range
        valid_indices = np.abs(lags/self.sample_rate) <= self.max_tau
        search_correlation = cc[valid_indices]

        search_lags = lags[valid_indices]
        
        # Find the highest peak
        max_idx = np.argmax(search_correlation)
        max_val = search_correlation[max_idx]
        max_lag = search_lags[max_idx]

        tau = max_lag / self.sample_rate
        
        # Calculate Confidence Metric
        std_cc = np.std(search_correlation)
        mean_cc = np.mean(search_correlation)
        z_score = (max_val - mean_cc) / std_cc
        p_value = 2 * (1 - sp.stats.norm.cdf(abs(z_score)))
        n_points = len(search_correlation)
        confidence = 1 - min(1.0, p_value * n_points / np.log(n_points))
        
        if return_correlation:
            return cc, tau, confidence
        else:
            return tau, confidence

    def tdoa_estimation(self, sig1, sig2):
        if self.method == "gcc":
            # Calculate max_tau based on the max distance between the microphones
            tdoa, confidence = self.gcc_phat(sig1, sig2)

        elif self.method == "corr":
            tdoa, confidence = self.correlation_tdoa(sig1, sig2)

        return tdoa, confidence

    def hyperbola(self, mic1_pos, mic2_pos, tdoa, points = 1000):
        """
        Draw hyperbola for given TDOA between two microphones using parametric equation.
        
        Args:
            mic1_pos: Position of first microphone [x,y]
            mic2_pos: Position of second microphone [x,y]
            tdoa: Time difference of arrival (seconds)
            c: Speed of sound (m/s)
            points: Number of points to generate
            
        Returns:
            xy_points: Array of points on the hyperbola
        """
        # Distance difference from TDOA
        d = tdoa * self.speed_of_sound
        
        # Focal points are the microphone positions
        f1 = mic1_pos
        f2 = mic2_pos
        
        # Calculate center and rotation
        center = (f1 + f2) / 2
        focal_dist = np.linalg.norm(f2 - f1)
        
        # Half the distance between foci
        a = abs(d) / 2
        
        # Calculate b using focal distance (2c)
        c_val = focal_dist / 2
        if abs(a) >= c_val:
            return [0], [0]  # No real solution
        b = np.sqrt(c_val**2 - a**2)
        
        # Rotation angle from x-axis
        theta = np.arctan2(f2[1] - f1[1], f2[0] - f1[0])
        
        # Parametric equation for hyperbola
        t = np.linspace(-3, 3, points)
        
        if d > 0:  # First mic is reference
            x = a * np.cosh(t)
            y = b * np.sinh(t)
        else:  # Second mic is reference
            x = -a * np.cosh(t)
            y = b * np.sinh(t)
        
        # Rotation matrix
        R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
        
        # Apply rotation and translation
        xy_points = np.vstack((x, y)).T @ R.T + center
        
        xvals = xy_points[:, 0]
        yvals = xy_points[:, 1]

        return xvals, yvals

    def plot_correlations(self, signals):
        """
        Plot the direct cross-correlation for each microphone pair.
        
        Args:
            signals (list of np.ndarray): List of signals from each microphone
        """
        fig, axs = plt.subplots(len(signals), 1, figsize=(8, 4*len(signals)))
        fig.suptitle('Direct Cross-Correlation')
        for i in range(len(signals)):
            j = i + 1 if i + 1 < len(signals) else 0
            cc, tau, confidence = self.correlation_tdoa(signals[i], signals[j], return_correlation = True)
            time = np.arange(-(len(signals[i]) - 1), len(signals[j])) / self.sample_rate
            valid_range = np.abs(time) <= self.max_tau

            axs[i].plot(time, cc, alpha=0.75, linewidth=0.5)
            axs[i].scatter(tau, max(cc[valid_range]), marker = "x", color = "red", alpha=0.75, linewidth=0.5)
            axs[i].axvline(-self.max_tau, color='orange', linestyle='--', alpha = 0.5)
            axs[i].axvline(self.max_tau, color='orange', linestyle='--', alpha = 0.5)
            axs[i].set_xlim(-1.5 * self.max_tau, 1.5 * self.max_tau)
            axs[i].set_ylabel('Correlation')
            axs[i].set_title(f'Pair {i}-{j} [confidence: {confidence:.4f}]')
        axs[-1].set_xlabel('Time [s]')

    def plot_gcc(self, signals):
        """
        Plot the GCC-PHAT result for each microphone pair.
        
        Args:
            signals (list of np.ndarray): List of signals from each microphone
        """
        fig, axs = plt.subplots(len(signals), 1, figsize=(8, 3.5*len(signals)))
        if len(signals) == 1:  # Handle case of single signal
            axs = [axs]
        fig.suptitle('GCC-PHAT Correlation', y=0.95)
        
        for i in range(len(signals)):
            j = i + 1 if i + 1 < len(signals) else 0
            cc, tau, confidence = self.gcc_phat(signals[i], signals[j], return_correlation = True)
            
            # Shift and plot correlation

            tspan = (len(cc) // 2) / self.sample_rate
            time = np.linspace(-tspan, tspan, len(cc))
            valid_range = np.abs(time) <= self.max_tau
            index_of_max_within_valid_range = np.argmax(cc[valid_range])

            # Calculate our confidence metric components for plotting purposes
            zero_crossings = np.where(np.diff(np.sign(cc[valid_range])))[0]
            index_of_valid_range_start = np.argmax(valid_range)
            # Find the index of the second highest peak that is at least one zero crossing away from the main peak
            sorted_indices = np.argsort(cc[valid_range])[::-1]
            for idx in sorted_indices:
                if idx == index_of_max_within_valid_range:
                    continue
                zero_crossings_between = np.logical_and(zero_crossings > min(idx, index_of_max_within_valid_range), zero_crossings < max(idx, index_of_max_within_valid_range))
                if np.any(zero_crossings_between):
                    second_peak_index = idx + index_of_valid_range_start
                    break
            noise_floor_percentile = np.percentile(cc[valid_range], self.confidence_percentile)
            
            axs[i].plot(time, cc, alpha=0.75, linewidth=0.5)
            axs[i].scatter(tau, cc[valid_range][index_of_max_within_valid_range], marker = "x", color = "red", alpha=0.95, linewidth=1, label = "Correlation Peak")
            axs[i].scatter(time[second_peak_index], cc[second_peak_index], marker = "x", color = "orange", alpha=0.95, linewidth=1, label = "Second Peak")
            axs[i].axvline(x=-self.max_tau, color='orange', linestyle='--', alpha = 0.7, label = "Maximum likely TDOA (τ)")
            axs[i].axhline(y=noise_floor_percentile, color='green', linestyle='--', alpha = 0.7, label = f"Noise Floor ({self.confidence_percentile:0.1f}%)")
            axs[i].legend(loc = "lower left")
            axs[i].axvline(x=self.max_tau, color='orange', linestyle='--', alpha = 0.7)
            axs[i].set_xlim(-1.5 * self.max_tau, 1.5 * self.max_tau)
            axs[i].set_ylabel('Correlation')
            axs[i].set_title(f'Pair {i}-{j} [confidence: {confidence:.4f}]')
        axs[-1].set_xlabel('Time [s]')

    def plot_signals(self, signals):
        """
        Plot the spectrogram of each signal.
        
        Args:
            signals (list of np.ndarray): List of signals from each microphone
        """
        fig, axs = plt.subplots(len(signals), 1, figsize=(8, 4*len(signals)))
        fig.suptitle('Spectrogram of Signals')
        for i, sig in enumerate(signals):
            fft_size = 2**13
            noverlap = int(fft_size * 0.90)
            pad_ratio = 4
            axs[i].specgram(sig, Fs=self.sample_rate, NFFT=fft_size, noverlap=noverlap, pad_to=fft_size*pad_ratio, cmap=plt.cm.jet, vmin=-100, vmax=-30)
            axs[i].set_ylim(0, 1000)
            axs[i].set_xlabel('Time [s]')
            axs[i].set_ylabel('Frequency [Hz]')
            axs[i].grid()

    def locate_using_tdoas(self, tdoas, confidences):
        """
        Calculate initial position estimate for 2D TDOA using hyperbolic intersections, range estimations, and median angle.
        
        Args:
            tdoas (np.ndarray): TDOA measurements:
                tdoas[0]: TDOA between sensor 0 and 1
                tdoas[1]: TDOA between sensor 1 and 2
                tdoas[2]: TDOA between sensor 2 and 0
            confidences (np.ndarray): Confidence in the TDOA measurements:
                confidences[0]: Confidence in the TDOA between sensor 0 and 1
                confidences[1]: Confidence in the TDOA between sensor 1 and 2
                confidences[2]: Confidence in the TDOA between sensor 2 and 0
        
        Returns:
        ndarray of shape (2,) containing estimated [x,y] target position
        """
        # Calculate centroid of sensors
        centroid = np.mean(self.mic_positions, axis=0)
        
        # Convert TDOA to distance differences
        distance_differences = tdoas * self.speed_of_sound
        
        # Find intersections of each pair of hyperbolas
        intersections = []
        
        # Pairs to check: (0,1), (1,2), (0,2)
        pairs = [(0,1,2), (1,2,0), (2,0,1)] # @Note: This is not a generalizable solution. It is only valid for 3 microphones.
        
        for i, j, k in pairs: # @TODO restructure this loop so that we do not duplicate effort finding solutions for the same hyperbola (pair of microphones)
            si = self.mic_positions[i]
            sj = self.mic_positions[j]
            sk = self.mic_positions[k]
            
            # Function defining the hyperbola
            def hyperbola(xy):
                x, y = xy
                # First hyperbola
                d0 = np.sqrt((x - si[0])**2 + (y - si[1])**2)
                d1 = np.sqrt((x - sj[0])**2 + (y - sj[1])**2)
                # Second hyperbola
                # dd = np.sqrt((x - sj[0])**2 + (y - sj[1])**2) # @Note: This is redundant, same as D2
                d2 = np.sqrt((x - sk[0])**2 + (y - sk[1])**2)
                
                return [d0 - d1 - distance_differences[i],
                        d1 - d2 - distance_differences[j]]
            
            # Try multiple initial guesses around the sensor pair
            for scale in [0.5, 1.0, 1.5]:
                midpoint = (si + sj) / 2
                perpendicular = np.array([-distance_differences[i]/2, scale * np.linalg.norm(si - sj)])
                guess1 = midpoint + perpendicular
                guess2 = midpoint - perpendicular
                
                for guess in [guess1, guess2]:
                    try:
                        solution = sp.optimize.fsolve(hyperbola, guess)
                        residual = hyperbola(solution)
                        if abs(residual[0]) < 1e-6 and abs(residual[1] < 1e-6):  # Check if solution is valid
                            intersections.append(solution)
                            continue
                    except Exception as e:
                        continue
        
        # If no intersections found, return centroid with zero confidence
        if not intersections:
            return centroid, 0.0
        
        # Convert intersections to angular positions relative to centroid
        angles = []
        for intersection in intersections:
            relative_pos = intersection - centroid
            angle = np.arctan2(relative_pos[1], relative_pos[0])
            angles.append((angle, intersection))
        
        # Find median angle and corresponding intersection
        sorted_angles = sorted(angles, key=lambda x: x[0])
        median_idx = len(sorted_angles) // 2
        median_angle, median_intersection = sorted_angles[median_idx]
        
        # Calculate and bound distance
        direction = median_intersection - centroid
        
        if self.max_distance is None:
            distance = np.linalg.norm(direction)
        else:
            distance = np.minimum(np.linalg.norm(direction), self.max_distance)
        
        # Calculate final position
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            position = centroid + direction * distance
        else:
            position = centroid
        
        if self.confidence_metric == "min":
            confidence = np.min(confidences)
        elif self.confidence_metric == "mean":
            confidence = np.mean(confidences)
        elif self.confidence_metric == "median":
            confidence = np.median(confidences)
        else:
            raise ValueError(f"Invalid confidence metric: {self.confidence_metric}")

        return position, confidence

    def locate_source(self, signals, debug = False):
        """
        Perform multilateration to localize the sound source.
        
        Args:
            signals (list of np.ndarray): List of signals from each microphone
            debug (bool, optional): Whether to plot the signals, correlations, and GCC-PHAT results (default is False)
        
        Returns:
            position (np.ndarray): Estimated position of the sound source [x,y]
            confidence (float [0,1]): Confidence in the localization estimate
        """
        num_mics = len(self.mic_positions)
        tdoas = np.zeros((num_mics))
        confidences = np.zeros((num_mics))
        for i in range(num_mics):
            j = i + 1 if i + 1 < num_mics else 0
            tdoas[i], confidences[i] = self.tdoa_estimation(signals[i], signals[j])

        position, confidence = self.locate_using_tdoas(tdoas, confidences)

        if debug:
            self.plot_signals(signals)
            if self.method == "corr":
                self.plot_correlations(signals)
            if self.method == "gcc":
                self.plot_gcc(signals)

            # Plot the localized source
            plt.figure(figsize=(8, 8))
            plt.plot(self.mic_positions[:, 0], self.mic_positions[:, 1], 'ro', markersize=10, markerfacecolor='none', label='Microphones')
            for i in range(num_mics):
                j = i + 1 if i + 1 < num_mics else 0
                xvals, yvals = self.hyperbola(self.mic_positions[i], self.mic_positions[j], tdoas[i])
                if i == 0:
                    plt.plot(xvals, yvals, 'b-', alpha=0.3, label=f'Hyperbolas')
                else:
                    plt.plot(xvals, yvals, 'b-', alpha=0.3)
                plt.gca().annotate(f'{i}-{j}', (xvals[0], yvals[0]))
            for i in range(len(self.mic_positions)):
                plt.gca().annotate(f'{i}', (self.mic_positions[i][0], self.mic_positions[i][1]))
            plt.plot(position[0], position[1], 'gx', markersize=15, label='Estimated Source')
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.title('Localization Result')
            plt.legend()
            plt.grid()
            plt.axis('equal')

        return position, confidence

    
if __name__ == "__main__":
    # Define the microphone positions and other parameters
    mic_positions = np.array([(-50, 20), (-10, -40), (40, 10)])
    sample_rate = 48000  # Sample rate in Hz
    speed_of_sound = 1480  # Speed of sound in water in m/s

    # Generate a test signal (chirp) and delayed versions for each microphone
    duration = 1.0  # Signal duration in seconds
    noise_duration = 1.0
    noise_power = 40.0 # Noise power in dB to add to the signal before estimating TDOAs
    t = np.linspace(0, duration, int(sample_rate * duration))
    f0 = 100  # Start frequency of the chirp
    f1 = 1000  # End frequency of the chirp
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))

    # Define the true source position (in the same coordinate system as the microphones)
    source_pos = np.array([100, -130])

    ### Sanity Check: TDOA Estimation ###
    # Calculate the expected TDOAs for each microphone pair based on the source position and the microphone positions
    expected_tdoas = []
    for i in range(len(mic_positions)):
        j = i + 1 if i + 1 < len(mic_positions) else 0
        mic_position_a = mic_positions[i,:]
        mic_position_b = mic_positions[j,:]
        da = np.sqrt(np.sum((mic_position_a[0] - source_pos[0])**2 + (mic_position_a[1] - source_pos[1])**2))
        db = np.sqrt(np.sum((mic_position_b[0] - source_pos[0])**2 + (mic_position_b[1] - source_pos[1])**2))
        # Equivalent:
        # expected_tdoas.append(np.linalg.norm(mic_positions[i] - source_pos) - np.linalg.norm(mic_positions[j] - source_pos) / speed_of_sound)
        expected_tdoas.append((da - db) / speed_of_sound)
    expected_tdoas = np.array(expected_tdoas)
    print(f"Expected TDOAs: {expected_tdoas}")
    mlat = Multilateration(mic_positions, sample_rate, speed_of_sound, max_distance = 1000, method = "corr")
    test_location, confidence = mlat.locate_using_tdoas(expected_tdoas, confidences = np.ones((len(expected_tdoas))))
    print(f"Expected Location: {test_location}, Confidence: {confidence:.6f}")
    assert np.allclose(test_location, source_pos)
    #########################################

    delays = []
    for microphone_index in range(len(mic_positions)):
        mic_position = mic_positions[microphone_index]
        d = np.sqrt(np.sum((mic_position[0] - source_pos[0])**2 + (mic_position[1] - source_pos[1])**2))
        delays.append(d / speed_of_sound)
    delays = np.array(delays)


    # Generate the delayed signals for each microphone
    signals = []
    for delay in delays:
        delayed_chirp = np.zeros_like(chirp)
        delayed_chirp[int(delay * sample_rate):] = chirp[:-int(delay * sample_rate)]
        s = np.hstack((np.zeros((int(noise_duration * sample_rate),)), delayed_chirp, np.zeros((int(noise_duration * sample_rate),))))
        signals.append(s)

    # Add random noise to the signals
    for i in range(len(signals)):
        noise = np.random.normal(0, np.sqrt(noise_power), len(signals[i]))
        signals[i] += noise
        # Renormalize the signal to be within the range of -1 to 1
        signals[i] = signals[i] / np.max(np.abs(signals[i]))

    for method in ["corr", "gcc"]:
        mlat = Multilateration(mic_positions, sample_rate, speed_of_sound, method = method, corr_smoothing = 21)
        estimated_tdoas = []
        for microphone_index in range(len(mic_positions)):
            j = microphone_index + 1 if microphone_index + 1 < len(mic_positions) else 0
            e_tdoa, _ = mlat.tdoa_estimation(signals[microphone_index], signals[j])
            estimated_tdoas.append(e_tdoa)
        print(f"Estimated TDOAs: {estimated_tdoas}")
        # Perform multilateration to estimate the source position
        estimated_source, confidence = mlat.locate_source(signals, debug=True)
        print(f'Estimated source position ({method}): {estimated_source}, Confidence: {confidence:.6f}')
        plt.scatter(source_pos[0], source_pos[1], color='black', label='True Source')
        plt.legend()
    plt.show()