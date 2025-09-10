"""
Advanced Signal Processing with Multiple Transform Methods
Target: 25-40% coverage, ~65 cyclomatic complexity
"""

import math
import time
from typing import List, Dict, Any, Optional, Tuple, Callable, Complex
from dataclasses import dataclass
from enum import Enum

class WindowType(Enum):
    HAMMING = "hamming"
    HANNING = "hanning"
    BLACKMAN = "blackman"
    KAISER = "kaiser"
    RECTANGULAR = "rectangular"

class FilterType(Enum):
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"
    NOTCH = "notch"

@dataclass
class SignalProperties:
    sample_rate: float
    length: int
    frequency_content: Dict[str, float]
    noise_level: float
    dynamic_range: float

class DigitalSignalProcessor:
    def __init__(self, sample_rate: float = 44100.0):
        self.sample_rate = sample_rate
        self.nyquist_frequency = sample_rate / 2.0
        self.processing_history: List[str] = []
        self.filter_coefficients: Dict[str, List[float]] = {}
        self.window_cache: Dict[Tuple[int, str], List[float]] = {}
        
    def process_signal(self, signal: List[float], operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply sequence of signal processing operations
        """
        if not self._validate_signal(signal):
            return {'error': 'invalid_signal'}
        
        if not operations or len(operations) > 20:
            return {'error': 'invalid_operations_count'}
        
        result_signal = signal.copy()
        operation_results = []
        
        for i, operation in enumerate(operations):
            if not isinstance(operation, dict) or 'type' not in operation:
                return {'error': f'invalid_operation_at_{i}'}
            
            op_type = operation['type']
            
            try:
                if op_type == 'fft':
                    result = self._apply_fft(result_signal, operation)
                elif op_type == 'filter':
                    result = self._apply_filter(result_signal, operation)
                elif op_type == 'window':
                    result = self._apply_window(result_signal, operation)
                elif op_type == 'resample':
                    result = self._apply_resampling(result_signal, operation)
                elif op_type == 'noise_reduction':
                    result = self._apply_noise_reduction(result_signal, operation)
                elif op_type == 'envelope_detection':
                    result = self._apply_envelope_detection(result_signal, operation)
                elif op_type == 'spectral_analysis':
                    result = self._apply_spectral_analysis(result_signal, operation)
                else:
                    return {'error': f'unknown_operation_{op_type}'}
                
                if 'error' in result:
                    return result
                
                result_signal = result['signal']
                operation_results.append(result)
                self.processing_history.append(f"{op_type}_{i}")
                
            except Exception as e:
                return {'error': f'operation_{op_type}_failed: {str(e)}'}
        
        # Compute final signal properties
        final_properties = self._compute_signal_properties(result_signal)
        
        return {
            'processed_signal': result_signal,
            'operation_results': operation_results,
            'signal_properties': final_properties,
            'processing_history': self.processing_history
        }
    
    def _validate_signal(self, signal: List[float]) -> bool:
        if not signal or len(signal) > 100000:
            return False
        
        for sample in signal:
            if not isinstance(sample, (int, float)) or math.isnan(sample) or math.isinf(sample):
                return False
        
        return True
    
    def _apply_fft(self, signal: List[float], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Fast Fourier Transform with windowing and zero-padding
        """
        n = len(signal)
        window_type = params.get('window', 'hanning')
        zero_padding = params.get('zero_padding', 0)
        overlap = params.get('overlap', 0.5)
        
        # Apply windowing if specified
        windowed_signal = signal.copy()
        if window_type != 'none':
            window = self._get_window(n, window_type)
            windowed_signal = [signal[i] * window[i] for i in range(n)]
        
        # Apply zero padding
        if zero_padding > 0:
            windowed_signal.extend([0.0] * zero_padding)
            n = len(windowed_signal)
        
        # Compute FFT using DFT (simplified implementation)
        fft_result = self._compute_dft(windowed_signal)
        
        # Compute magnitude and phase spectra
        magnitudes = [abs(complex_val) for complex_val in fft_result]
        phases = [math.atan2(complex_val.imag, complex_val.real) if complex_val.real != 0 else 0 
                 for complex_val in fft_result]
        
        # Compute frequency bins
        freq_bins = [i * self.sample_rate / n for i in range(n // 2)]
        
        # Find dominant frequencies
        dominant_freqs = []
        threshold = max(magnitudes) * 0.1 if magnitudes else 0
        
        for i in range(1, len(magnitudes) // 2):
            if (magnitudes[i] > threshold and 
                magnitudes[i] > magnitudes[i-1] and 
                magnitudes[i] > magnitudes[i+1]):
                dominant_freqs.append((freq_bins[i], magnitudes[i]))
        
        # Sort by magnitude
        dominant_freqs.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'signal': signal,  # FFT doesn't modify time domain signal
            'fft_magnitudes': magnitudes[:n//2],
            'fft_phases': phases[:n//2],
            'frequency_bins': freq_bins,
            'dominant_frequencies': dominant_freqs[:10],  # Top 10
            'spectral_centroid': self._compute_spectral_centroid(magnitudes, freq_bins)
        }
    
    def _apply_filter(self, signal: List[float], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply digital filter (IIR or FIR)
        """
        filter_type = params.get('filter_type', 'lowpass')
        cutoff_freq = params.get('cutoff_frequency', 1000.0)
        order = params.get('order', 4)
        design_method = params.get('design_method', 'butterworth')
        
        # Validate parameters
        if cutoff_freq >= self.nyquist_frequency:
            return {'error': 'cutoff_frequency_too_high'}
        
        if order <= 0 or order > 20:
            return {'error': 'invalid_filter_order'}
        
        # Generate filter coefficients
        if design_method == 'butterworth':
            coefficients = self._design_butterworth_filter(filter_type, cutoff_freq, order)
        elif design_method == 'chebyshev':
            ripple = params.get('ripple', 1.0)
            coefficients = self._design_chebyshev_filter(filter_type, cutoff_freq, order, ripple)
        elif design_method == 'fir':
            coefficients = self._design_fir_filter(filter_type, cutoff_freq, order)
        else:
            return {'error': 'unknown_filter_design_method'}
        
        if not coefficients:
            return {'error': 'filter_design_failed'}
        
        # Apply filter
        filtered_signal = self._apply_filter_coefficients(signal, coefficients)
        
        # Compute filter response
        frequency_response = self._compute_frequency_response(coefficients, 100)
        
        return {
            'signal': filtered_signal,
            'filter_coefficients': coefficients,
            'frequency_response': frequency_response,
            'filter_delay': len(coefficients) // 2,
            'effective_bandwidth': self._compute_effective_bandwidth(frequency_response)
        }
    
    def _apply_window(self, signal: List[float], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply windowing function to signal
        """
        window_type = params.get('window_type', 'hanning')
        window_length = params.get('length', len(signal))
        overlap = params.get('overlap', 0)
        
        if window_length > len(signal):
            window_length = len(signal)
        
        window = self._get_window(window_length, window_type)
        
        # Apply window with overlap processing if specified
        if overlap > 0:
            return self._apply_overlapping_windows(signal, window, overlap)
        else:
            windowed_signal = []
            for i in range(len(signal)):
                if i < window_length:
                    windowed_signal.append(signal[i] * window[i])
                else:
                    windowed_signal.append(signal[i])
            
            return {
                'signal': windowed_signal,
                'window_function': window,
                'coherent_gain': sum(window) / len(window),
                'processing_gain': math.sqrt(sum(w*w for w in window) / len(window))
            }
    
    def _apply_resampling(self, signal: List[float], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resample signal to different sample rate
        """
        target_rate = params.get('target_rate', self.sample_rate)
        method = params.get('method', 'linear')
        anti_alias = params.get('anti_alias', True)
        
        if target_rate <= 0:
            return {'error': 'invalid_target_rate'}
        
        ratio = target_rate / self.sample_rate
        
        # Apply anti-aliasing filter if downsampling
        if ratio < 1.0 and anti_alias:
            # Design anti-aliasing filter
            cutoff = target_rate / 2.0
            aa_coefficients = self._design_fir_filter('lowpass', cutoff, 64)
            signal = self._apply_filter_coefficients(signal, aa_coefficients)
        
        # Perform resampling
        if method == 'linear':
            resampled = self._linear_interpolation_resample(signal, ratio)
        elif method == 'cubic':
            resampled = self._cubic_interpolation_resample(signal, ratio)
        elif method == 'sinc':
            resampled = self._sinc_interpolation_resample(signal, ratio)
        else:
            return {'error': 'unknown_resampling_method'}
        
        return {
            'signal': resampled,
            'original_rate': self.sample_rate,
            'target_rate': target_rate,
            'resampling_ratio': ratio,
            'length_change': len(resampled) - len(signal)
        }
    
    def _apply_noise_reduction(self, signal: List[float], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply noise reduction algorithms
        """
        method = params.get('method', 'spectral_subtraction')
        noise_threshold = params.get('noise_threshold', 0.1)
        reduction_factor = params.get('reduction_factor', 0.5)
        
        if method == 'spectral_subtraction':
            return self._spectral_subtraction(signal, noise_threshold, reduction_factor)
        elif method == 'wiener_filter':
            return self._wiener_filtering(signal, params)
        elif method == 'adaptive_filter':
            return self._adaptive_filtering(signal, params)
        else:
            return {'error': 'unknown_noise_reduction_method'}
    
    def _apply_envelope_detection(self, signal: List[float], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract signal envelope using various methods
        """
        method = params.get('method', 'hilbert')
        smoothing = params.get('smoothing', 0.1)
        
        if method == 'hilbert':
            envelope = self._hilbert_envelope(signal)
        elif method == 'peak_detection':
            envelope = self._peak_envelope(signal, params)
        elif method == 'rms':
            window_size = params.get('window_size', 256)
            envelope = self._rms_envelope(signal, window_size)
        else:
            return {'error': 'unknown_envelope_method'}
        
        # Apply smoothing if requested
        if smoothing > 0:
            envelope = self._smooth_signal(envelope, smoothing)
        
        return {
            'signal': signal,  # Original signal unchanged
            'envelope': envelope,
            'envelope_statistics': {
                'mean': sum(envelope) / len(envelope),
                'max': max(envelope),
                'min': min(envelope),
                'variance': self._compute_variance(envelope)
            }
        }
    
    def _apply_spectral_analysis(self, signal: List[float], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform detailed spectral analysis
        """
        analysis_type = params.get('analysis_type', 'power_spectrum')
        window_size = params.get('window_size', 1024)
        overlap = params.get('overlap', 0.5)
        
        if analysis_type == 'power_spectrum':
            return self._power_spectrum_analysis(signal, window_size, overlap)
        elif analysis_type == 'spectrogram':
            return self._spectrogram_analysis(signal, window_size, overlap)
        elif analysis_type == 'cepstrum':
            return self._cepstrum_analysis(signal)
        elif analysis_type == 'autocorrelation':
            return self._autocorrelation_analysis(signal, params)
        else:
            return {'error': 'unknown_spectral_analysis_type'}
    
    # Helper methods for signal processing operations
    def _get_window(self, length: int, window_type: str) -> List[float]:
        """Generate window function"""
        cache_key = (length, window_type)
        if cache_key in self.window_cache:
            return self.window_cache[cache_key]
        
        window = []
        
        for i in range(length):
            if window_type == 'hamming':
                w = 0.54 - 0.46 * math.cos(2 * math.pi * i / (length - 1))
            elif window_type == 'hanning':
                w = 0.5 - 0.5 * math.cos(2 * math.pi * i / (length - 1))
            elif window_type == 'blackman':
                w = (0.42 - 0.5 * math.cos(2 * math.pi * i / (length - 1)) + 
                     0.08 * math.cos(4 * math.pi * i / (length - 1)))
            elif window_type == 'kaiser':
                beta = 8.6
                w = self._modified_bessel_i0(beta * math.sqrt(1 - ((2*i/(length-1)) - 1)**2)) / self._modified_bessel_i0(beta)
            else:  # rectangular
                w = 1.0
            
            window.append(w)
        
        self.window_cache[cache_key] = window
        return window
    
    def _compute_dft(self, signal: List[float]) -> List[complex]:
        """Compute Discrete Fourier Transform"""
        n = len(signal)
        dft_result = []
        
        for k in range(n):
            real_sum = 0.0
            imag_sum = 0.0
            
            for n_idx in range(n):
                angle = -2 * math.pi * k * n_idx / n
                real_sum += signal[n_idx] * math.cos(angle)
                imag_sum += signal[n_idx] * math.sin(angle)
            
            dft_result.append(complex(real_sum, imag_sum))
        
        return dft_result
    
    def _compute_spectral_centroid(self, magnitudes: List[float], freq_bins: List[float]) -> float:
        """Compute spectral centroid (brightness measure)"""
        if not magnitudes or not freq_bins:
            return 0.0
        
        weighted_sum = sum(freq_bins[i] * magnitudes[i] for i in range(min(len(freq_bins), len(magnitudes))))
        magnitude_sum = sum(magnitudes[:len(freq_bins)])
        
        return weighted_sum / magnitude_sum if magnitude_sum > 0 else 0.0
    
    def _design_butterworth_filter(self, filter_type: str, cutoff: float, order: int) -> List[float]:
        """Design Butterworth filter coefficients"""
        # Simplified implementation - returns basic coefficients
        coefficients = []
        
        # Normalize cutoff frequency
        wc = cutoff / self.nyquist_frequency
        
        if filter_type == 'lowpass':
            for i in range(order + 1):
                coef = math.exp(-i * wc)
                coefficients.append(coef)
        elif filter_type == 'highpass':
            for i in range(order + 1):
                coef = math.exp(-i * wc) * (-1) ** i
                coefficients.append(coef)
        else:
            # Default to simple lowpass
            coefficients = [1.0, 0.5, 0.25]
        
        # Normalize
        coef_sum = sum(abs(c) for c in coefficients)
        if coef_sum > 0:
            coefficients = [c / coef_sum for c in coefficients]
        
        return coefficients
    
    def _apply_filter_coefficients(self, signal: List[float], coefficients: List[float]) -> List[float]:
        """Apply filter coefficients using convolution"""
        filtered = []
        coef_len = len(coefficients)
        
        for i in range(len(signal)):
            output = 0.0
            
            for j, coef in enumerate(coefficients):
                if i - j >= 0:
                    output += coef * signal[i - j]
            
            filtered.append(output)
        
        return filtered
    
    def _compute_signal_properties(self, signal: List[float]) -> SignalProperties:
        """Compute comprehensive signal properties"""
        if not signal:
            return SignalProperties(self.sample_rate, 0, {}, 0.0, 0.0)
        
        # Basic statistics
        mean_val = sum(signal) / len(signal)
        variance = sum((x - mean_val) ** 2 for x in signal) / len(signal)
        rms_val = math.sqrt(sum(x * x for x in signal) / len(signal))
        
        # Dynamic range
        max_val = max(signal)
        min_val = min(signal)
        dynamic_range = 20 * math.log10(max_val / max(abs(min_val), 1e-10)) if max_val > 0 else 0
        
        # Estimate noise level (high frequency content)
        noise_estimate = math.sqrt(variance) / (rms_val + 1e-10)
        
        # Basic frequency content estimation
        freq_content = {
            'dc_component': abs(mean_val),
            'ac_component': rms_val,
            'total_energy': sum(x * x for x in signal),
            'peak_factor': max_val / (rms_val + 1e-10)
        }
        
        return SignalProperties(
            sample_rate=self.sample_rate,
            length=len(signal),
            frequency_content=freq_content,
            noise_level=noise_estimate,
            dynamic_range=dynamic_range
        )
    
    def _modified_bessel_i0(self, x: float) -> float:
        """Modified Bessel function of first kind, order 0"""
        ax = abs(x)
        if ax < 3.75:
            y = (x / 3.75) ** 2
            return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + 
                   y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))))
        else:
            z = 3.75 / ax
            return (math.exp(ax) / math.sqrt(ax)) * (0.39894228 + z * (0.1328592e-1 + 
                   z * (0.225319e-2 + z * (-0.157565e-2 + z * (0.916281e-2 + 
                   z * (-0.2057706e-1 + z * (0.2635537e-1 + z * (-0.1647633e-1 + 
                   z * 0.392377e-2))))))))
    
    # Additional helper methods (simplified implementations)
    def _linear_interpolation_resample(self, signal: List[float], ratio: float) -> List[float]:
        new_length = int(len(signal) * ratio)
        resampled = []
        
        for i in range(new_length):
            old_index = i / ratio
            idx = int(old_index)
            frac = old_index - idx
            
            if idx + 1 < len(signal):
                value = signal[idx] * (1 - frac) + signal[idx + 1] * frac
            else:
                value = signal[idx] if idx < len(signal) else 0.0
            
            resampled.append(value)
        
        return resampled
    
    def _spectral_subtraction(self, signal: List[float], threshold: float, reduction: float) -> Dict[str, Any]:
        # Simplified spectral subtraction
        fft_result = self._compute_dft(signal)
        
        for i in range(len(fft_result)):
            magnitude = abs(fft_result[i])
            if magnitude < threshold:
                fft_result[i] = complex(fft_result[i].real * reduction, fft_result[i].imag * reduction)
        
        # Convert back (simplified - would need IFFT)
        processed_signal = [abs(x) for x in fft_result]
        
        return {
            'signal': processed_signal[:len(signal)],
            'noise_reduction_applied': reduction,
            'threshold_used': threshold
        }

def signal_processor_original(input_signal: List[float], processing_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Target function for test case generation with high complexity
    """
    # Validate inputs
    if not isinstance(input_signal, list) or not input_signal:
        return {'error': 'invalid_input_signal'}
    
    if len(input_signal) > 50000:
        return {'error': 'signal_too_long'}
    
    if not isinstance(processing_config, dict):
        return {'error': 'invalid_processing_config'}
    
    # Extract configuration
    sample_rate = processing_config.get('sample_rate', 44100.0)
    operations = processing_config.get('operations', [])
    
    if sample_rate <= 0 or sample_rate > 192000:
        return {'error': 'invalid_sample_rate'}
    
    if not operations or len(operations) > 10:
        return {'error': 'invalid_operations_list'}
    
    # Validate signal values
    for i, sample in enumerate(input_signal):
        if not isinstance(sample, (int, float)):
            return {'error': f'invalid_sample_at_{i}'}
        
        if abs(sample) > 100:  # Reasonable amplitude limit
            return {'error': 'signal_amplitude_too_high'}
    
    # Create processor and process signal
    processor = DigitalSignalProcessor(sample_rate)
    
    try:
        result = processor.process_signal(input_signal, operations)
        
        if 'error' in result:
            return result
        
        # Analyze processing complexity
        processed_signal = result['processed_signal']
        properties = result['signal_properties']
        
        # Determine processing outcome
        if len(processed_signal) == 0:
            status = 'signal_eliminated'
        elif properties.dynamic_range > 60:
            status = 'high_dynamic_range'
        elif properties.noise_level > 0.5:
            status = 'noisy_output'
        elif len(result['operation_results']) != len(operations):
            status = 'partial_processing'
        elif any('error' in op_result for op_result in result['operation_results']):
            status = 'processing_errors'
        else:
            # Complex decision based on signal characteristics
            energy_ratio = properties.frequency_content['total_energy'] / max(sum(x*x for x in input_signal), 1)
            
            if energy_ratio > 2.0:
                status = 'signal_amplified'
            elif energy_ratio < 0.1:
                status = 'signal_attenuated'
            elif abs(energy_ratio - 1.0) < 0.1:
                status = 'signal_preserved'
            else:
                status = 'signal_modified'
        
        return {
            'status': status,
            'processing_result': result,
            'input_length': len(input_signal),
            'output_length': len(processed_signal),
            'operations_applied': len(operations)
        }
        
    except Exception as e:
        return {'error': f'processing_exception: {str(e)}'}


def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original target_function to provide
    a consistent 4-parameter interface for automated test generation.
    
    Original function: target_function(input_signal, processing_config)
    
    Args:
        a: First parameter (required)
        b: Second parameter (optional)
        c: Third parameter (optional) 
        d: Fourth parameter (optional)
    
    Returns:
        Result from target_function
    """
    # Set defaults for optional parameters based on function requirements
    if c is None:
        c = 0
    if d is None:
        d = 20
    if b is None:
        b = -20
    if c is None:
        c = 0
    if d is None:
        d = 20
    
    # Validate parameter ranges
    for param, name in [(a, 'a'), (b, 'b'), (c, 'c'), (d, 'd')]:
        if param is not None and isinstance(param, (int, float)):
            if not (-20 <= param <= 20):
                param = max(-20, min(20, param))  # Clamp to bounds
    
    # Call original function with appropriate parameters
    try:
        return target_function(a, b)
    except Exception as e:
        # Handle potential errors gracefully for test generation
        if "too many" in str(e).lower() or "unexpected keyword" in str(e).lower():
            # Try with fewer parameters
            try:
                return target_function(a)
            except:
                pass
            return target_function(a)
        raise e
