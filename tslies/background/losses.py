import tensorflow as tf
import tf_keras
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

@tf_keras.utils.register_keras_serializable()
class CustomLosses:
    def __init__(self, normalizations_dict):
        """The class for custom loss functions."""
        self.normalizations_dict = normalizations_dict

    def get_config(self):
        """Returns the configuration of the custom loss functions."""
        pass

    @tf_keras.utils.register_keras_serializable()
    def mae(self, y_true, y_pred):
        """The mean absolute error metric computed using the mean of the output."""
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    @tf_keras.utils.register_keras_serializable()
    def mae_bnn(self, y_true, y_pred):
        """The mean absolute error metric computed using the mean of the output."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        return tf.reduce_mean(tf.abs(y_true - mean)) / self.normalizations_dict['mae']

    @tf_keras.utils.register_keras_serializable()
    def negative_log_likelihood_stddev(self, y_true, y_pred):
        """The negative log likelihood loss function computed using the mean and standard deviation."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        var = y_pred[:, num_outputs:]
        norm_dist = tfd.Normal(loc=mean, scale=1e-3 + tf.math.softplus(0.05*var))
        nll = -norm_dist.log_prob(y_true)
        return nll

    @tf_keras.utils.register_keras_serializable()
    def negative_log_likelihood(self, y_true, y_pred):
        """The negative log likelihood loss function computed using the mean and variance."""
        mean = y_pred.mean()
        log_var = y_pred.variance()
        nll = 0.5 * log_var + 0.5 * tf.square(y_true - mean) * tf.exp(-log_var)
        return tf.reduce_mean(nll)
    
    @tf_keras.utils.register_keras_serializable()
    def negative_log_likelihood_bnn(self, y_true, y_pred):
        """The negative log likelihood loss function computed using the mean and variance."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        stddev = y_pred[:, num_outputs:]
        nll = 0.5 * tf.square((y_true - mean) / stddev) + tf.math.log(stddev)
        
        return tf.reduce_mean(nll)

    @tf_keras.utils.register_keras_serializable()
    def negative_log_likelihood_var(self, y_true, y_pred):
        """The negative log likelihood loss function computed using the mean and variance."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        log_var = y_pred[:, num_outputs:]
        
        # Clamp log_var to prevent numerical instability
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        
        # Use numerically stable formulation
        nll = 0.5 * log_var + 0.5 * tf.square(y_true - mean) * tf.exp(-log_var)
        return tf.reduce_mean(nll)

    @tf_keras.utils.register_keras_serializable()
    def negative_log_likelihood_spectral(self, y_true, y_pred):
        """The negative log likelihood loss function computed using the mean and variance."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        log_var = y_pred[:, num_outputs:]
        
        # Clamp log_var to prevent numerical instability
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        
        # Use numerically stable formulation
        nll = 0.5 * log_var + 0.5 * tf.square(y_true - mean) * tf.exp(-log_var)
        return tf.reduce_mean(nll) + self.spectral_loss(y_true, mean)

    @tf_keras.utils.register_keras_serializable()
    def negative_log_likelihood_abs(self, y_true, y_pred):
        """The negative log likelihood loss function computed using the mean and variance."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        log_var = y_pred[:, num_outputs:]
        
        # Clamp log_var to prevent numerical instability
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        
        # Use numerically stable formulation
        nll = 0.5 * log_var + 0.5 * tf.abs(y_true - mean) * tf.exp(-log_var)
        return tf.reduce_mean(nll)
    
    def spectral_loss_bnn(self, y_true, y_pred):
        """Spectral loss for Bayesian Neural Networks."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        return self.spectral_loss(y_true, mean)

    @tf_keras.utils.register_keras_serializable()
    def aic(self, y_true, y_pred):
        """The Akaike Information Criterion loss function."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        log_var = y_pred[:, num_outputs:]
        nll = 0.5 * log_var + 0.5 * tf.square(y_true - mean) * tf.exp(-log_var)
        return tf.reduce_mean(nll) + 2 * num_outputs

    @tf_keras.utils.register_keras_serializable()
    def negative_log_likelihood_huber(self, y_true, y_pred, delta=0.003):
        """Gaussian negative log-likelihood with a Huber loss for robustness."""
        num_outputs = tf.shape(y_true)[1]
        mean = y_pred[:, :num_outputs]
        log_var = y_pred[:, num_outputs:]
        
        # huber loss
        residual = tf.abs(y_true - mean)
        huber_loss = tf.where(residual <= delta, 0.5 * tf.square(residual), delta * (residual - 0.5 * delta))
        nll = 0.5 * log_var + huber_loss * tf.exp(-log_var)
        return tf.reduce_mean(nll)

    @tf_keras.utils.register_keras_serializable()
    def mse(self, y_true, y_pred):
        """The mean squared error metric computed using the mean of the output."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        return tf.reduce_mean(tf.square(y_true - mean)) / self.normalizations_dict['mse']

    @tf_keras.utils.register_keras_serializable()
    def r_squared(self, y_true, y_pred):
        """The R-squared metric computed using the mean of the output."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        ss_res = tf.reduce_sum(tf.square(y_true - mean))
        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=0)))
        return 1 - ss_res / (ss_tot + tf_keras.backend.epsilon())

    @tf_keras.utils.register_keras_serializable()
    def NLL(self, y_true, y_pred):
        """The negative log likelihood loss function computed using the mean and variance."""
        return -y_pred.log_prob(y_true)

    @tf_keras.utils.register_keras_serializable()
    def nll_metric(self, y_true, y_pred):
        """The negative log likelihood loss function computed using the mean and variance."""
        mean = tf.reduce_mean(y_pred)
        log_var = tf.math.log(tf.math.reduce_variance(y_pred))
        nll = 0.5 * log_var + 0.5 * tf.square(y_true - mean) * tf.exp(-log_var)
        return tf.reduce_mean(nll)

    @tf_keras.utils.register_keras_serializable()
    def NLL_median(self, y_true, y_pred):
        """The negative log likelihood loss function computed using the mean and variance."""
        return tfp.stats.percentile(y_pred.log_prob(y_true),50., interpolation='midpoint')

    @tf_keras.utils.register_keras_serializable()
    def MSE(self, y_true, y_pred):
        """The mean squared error loss function computed using the mean of the output."""
        return tf.reduce_mean(tf.square(y_true - y_pred))

    @tf_keras.utils.register_keras_serializable()
    def MAE(self, y_true, y_pred):
        """The mean absolute error loss function computed using the mean of the output."""
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    @tf_keras.utils.register_keras_serializable()
    def KL_divergence(self, posterior, prior):
        """The Kullback-Leibler divergence loss function."""
        return tfp.distributions.kl_divergence(posterior, prior)
    
    @tf_keras.utils.register_keras_serializable()
    def spectral_loss(self, y_true, y_pred):
        y_true_fft = tf.signal.rfft(y_true)
        y_pred_fft = tf.signal.rfft(y_pred)
        diff_fft = y_true_fft - y_pred_fft
        spectral_loss = tf.reduce_mean(tf.square(tf.abs(diff_fft)))
        return spectral_loss

    @tf_keras.utils.register_keras_serializable()
    def balanced_uncertainty_loss(self, y_true, y_pred, alpha=1.0, beta=0.1):
        """Balanced loss that prevents uncertainty collapse while maintaining accuracy."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        log_var = y_pred[:, num_outputs:]
        
        # Clamp log_var to prevent numerical instability
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        
        # Standard NLL term
        nll = 0.5 * log_var + 0.5 * tf.square(y_true - mean) * tf.exp(-log_var)
        
        # Uncertainty regularization that prevents collapse
        # Penalize both too high and too low uncertainty
        target_log_var = tf.math.log(tf.reduce_mean(tf.square(y_true - mean)) + 1e-8)
        uncertainty_penalty = tf.square(log_var - target_log_var)
        
        return alpha * tf.reduce_mean(nll) + beta * tf.reduce_mean(uncertainty_penalty)

    @tf_keras.utils.register_keras_serializable()
    def evidential_loss(self, y_true, y_pred, coeff=1.0):
        """Evidential loss that learns uncertainty without collapse."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        log_var = y_pred[:, num_outputs:]
        
        # Clamp log_var
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        var = tf.exp(log_var)
        
        # Evidential terms
        alpha = var + 1e-8  # Evidence parameter
        
        # NLL term
        diff = y_true - mean
        nll = 0.5 * tf.math.log(2 * 3.14159265359 * alpha) + 0.5 * tf.square(diff) / alpha
        
        # Regularization term that encourages higher uncertainty for larger errors
        reg = tf.abs(diff) * alpha
        
        return tf.reduce_mean(nll + coeff * reg)

    @tf_keras.utils.register_keras_serializable()
    def adaptive_uncertainty_loss(self, y_true, y_pred, lambda_adapt=0.1):
        """Adaptive loss that adjusts uncertainty based on prediction quality."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        log_var = y_pred[:, num_outputs:]
        
        # Clamp log_var
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        
        # Compute residuals
        residuals = tf.abs(y_true - mean)
        
        # Adaptive weight: higher weight for samples with large residuals
        adaptive_weights = 1.0 + lambda_adapt * residuals / (tf.reduce_mean(residuals) + 1e-8)
        
        # Weighted NLL
        nll = 0.5 * log_var + 0.5 * tf.square(y_true - mean) * tf.exp(-log_var)
        weighted_nll = adaptive_weights * nll
        
        return tf.reduce_mean(weighted_nll)

    @tf_keras.utils.register_keras_serializable()
    def heteroscedastic_loss(self, y_true, y_pred, min_var=1e-6):
        """Heteroscedastic loss with minimum variance constraint."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        log_var = y_pred[:, num_outputs:]
        
        # Clamp log_var with a minimum variance
        log_var = tf.clip_by_value(log_var, tf.math.log(min_var), 10.0)
        
        # Standard NLL with minimum variance constraint
        nll = 0.5 * log_var + 0.5 * tf.square(y_true - mean) * tf.exp(-log_var)
        
        # Add penalty for extremely low uncertainty in easy regions
        # This prevents the model from being overconfident
        overconfidence_penalty = tf.reduce_mean(tf.maximum(0.0, -log_var - tf.math.log(min_var * 10)))
        
        return tf.reduce_mean(nll) + 0.01 * overconfidence_penalty

    @tf_keras.utils.register_keras_serializable()
    def kl_divergence_loss(self, y_true, y_pred):
        """The Kullback-Leibler divergence loss function."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        stddev = y_pred[:, num_outputs:]
        prior = tfd.Normal(loc=0., scale=1.)
        posterior = tfd.Normal(loc=mean, scale=1e-3 + tf.math.softplus(0.05*stddev))
        kl_div = tfp.distributions.kl_divergence(posterior, prior)
        return tf.reduce_mean(kl_div)

    @tf_keras.utils.register_keras_serializable()
    def mean_uncert(self, y_true, y_pred):
        """Mean predicted uncertainty (standard deviation) across all outputs."""
        num_outputs = y_true.shape[1]
        log_var = y_pred[:, num_outputs:]
        
        # Clamp log_var to prevent numerical instability
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        uncertainty = tf.sqrt(tf.exp(log_var))
        return tf.reduce_mean(uncertainty)

    @tf_keras.utils.register_keras_serializable()
    def uncert_mae_ratio(self, y_true, y_pred):
        """Ratio between mean uncertainty and MAE - lower is better."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        log_var = y_pred[:, num_outputs:]
        
        # Clamp log_var to prevent numerical instability
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        
        mae = tf.reduce_mean(tf.abs(y_true - mean))
        uncertainty = tf.reduce_mean(tf.sqrt(tf.exp(log_var)))
        
        # Add small epsilon to prevent division by zero
        return uncertainty / (mae + 1e-8)

    @tf_keras.utils.register_keras_serializable()
    def calibration_error(self, y_true, y_pred):
        """Calibration error - measures if predicted uncertainties match actual errors."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        log_var = y_pred[:, num_outputs:]
        
        # Clamp log_var to prevent numerical instability
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        
        errors = tf.abs(y_true - mean)
        predicted_std = tf.sqrt(tf.exp(log_var))
        
        # Normalized errors (should follow standard normal if well calibrated)
        normalized_errors = errors / (predicted_std + 1e-8)
        
        # Ideal mean for normalized errors should be around sqrt(2/pi) ≈ 0.798
        ideal_mean = tf.sqrt(2.0 / tf.constant(3.14159265359))
        calibration_error = tf.abs(tf.reduce_mean(normalized_errors) - ideal_mean)
        
        return calibration_error

    @tf_keras.utils.register_keras_serializable()
    def prediction_interval_coverage(self, y_true, y_pred, confidence=0.68):
        """Coverage probability for prediction intervals (should match confidence level)."""
        num_outputs = y_true.shape[1]
        mean = y_pred[:, :num_outputs]
        log_var = y_pred[:, num_outputs:]
        
        # Clamp log_var to prevent numerical instability
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        std = tf.sqrt(tf.exp(log_var))
        
        # For 68% confidence interval (1 sigma)
        z_score = tf.constant(1.0) if confidence == 0.68 else tf.constant(1.96)  # 95%
        
        lower_bound = mean - z_score * std
        upper_bound = mean + z_score * std
        
        # Check if true values fall within prediction interval
        within_interval = tf.logical_and(y_true >= lower_bound, y_true <= upper_bound)
        coverage = tf.reduce_mean(tf.cast(within_interval, tf.float32))
        
        return coverage

    @tf_keras.utils.register_keras_serializable()
    def sharpness_metric(self, y_true, y_pred):
        """Sharpness of prediction intervals - lower uncertainty is sharper."""
        num_outputs = y_true.shape[1]
        log_var = y_pred[:, num_outputs:]
        
        # Clamp log_var to prevent numerical instability
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        
        # Average width of 95% prediction intervals
        std = tf.sqrt(tf.exp(log_var))
        interval_width = 2 * 1.96 * std  # 95% interval width
        
        return tf.reduce_mean(interval_width)
        
    @tf_keras.utils.register_keras_serializable()
    def max_uncert(self, y_true, y_pred):
        """Maximum predicted uncertainty (standard deviation) across all outputs."""
        num_outputs = y_true.shape[1]
        log_var = y_pred[:, num_outputs:]
        
        # Clamp log_var to prevent numerical instability
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        uncertainty = tf.sqrt(tf.exp(log_var))
        return tf.reduce_max(uncertainty)
    
    @tf_keras.utils.register_keras_serializable()
    def min_uncert(self, y_true, y_pred):
        """Minimum predicted uncertainty (standard deviation) across all outputs."""
        num_outputs = y_true.shape[1]
        log_var = y_pred[:, num_outputs:]
        
        # Clamp log_var to prevent numerical instability
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        uncertainty = tf.sqrt(tf.exp(log_var))
        return tf.reduce_min(uncertainty)
    
    def get_loss_list(self):
        """Returns a dictionary of available loss functions."""
        return {
            'mae': self.mae,
            'mae_bnn': self.mae_bnn,
            'mse': self.mse,
            'r_squared': self.r_squared,
            'negative_log_likelihood_stddev': self.negative_log_likelihood_stddev,
            'negative_log_likelihood': self.negative_log_likelihood,
            'negative_log_likelihood_bnn': self.negative_log_likelihood_bnn,
            'negative_log_likelihood_var': self.negative_log_likelihood_var,
            'aic': self.aic,
            'negative_log_likelihood_huber': self.negative_log_likelihood_huber,
            'NLL': self.NLL,
            'NLL_median': self.NLL_median,
            'MSE': self.MSE,
            'MAE': self.MAE,
            'KL_divergence': self.KL_divergence,
            'kl_divergence_loss': self.kl_divergence_loss,
            'spectral_loss': self.spectral_loss,
            'mean_uncert': self.mean_uncert,
            'uncert_mae_ratio': self.uncert_mae_ratio,
            'calibration_error': self.calibration_error,
            'prediction_interval_coverage': self.prediction_interval_coverage,
            'sharpness_metric': self.sharpness_metric,
            'max_uncert': self.max_uncert,
            'min_uncert': self.min_uncert,
            'balanced_uncertainty_loss': self.balanced_uncertainty_loss,
            'evidential_loss': self.evidential_loss,
            'adaptive_uncertainty_loss': self.adaptive_uncertainty_loss,
            'heteroscedastic_loss': self.heteroscedastic_loss,
            'spectral_loss_bnn': self.spectral_loss_bnn,
            'negative_log_likelihood_spectral': self.negative_log_likelihood_spectral
        }
    
    def get_metrics_list(self):
        """Returns a dictionary of available metrics."""
        return self.get_loss_list()