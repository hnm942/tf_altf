import tensorflow as tf

class RandomInterpolation:
    def __init__(self, scale_range, shift_range):
        # create scale random value rate
        self.scale_factor = tf.random.uniform([], minval= scale_range[0], maxval= scale_range[1])
        # create shift random value rate
        self.shift_factor = tf.cast(tf.random.uniform([], minval= shift_range[0], maxval= shift_range[1]), dtype=tf.int32)
    def scale_and_shift(self, data):
        scale_matrix = tf.eye(tf.shape(data)[0]) * self.scale_factor
        
        # Dịch chuyển dữ liệu
        shifted_data = tf.roll(data, shift=self.shift_factor, axis=1)
        interpolation_data = tf.matmul(scale_matrix, shifted_data)
        return interpolation_data
    
