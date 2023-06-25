import tensorflow as tf

class Translation():
    def __init__(self, translation_vector):
        self.translation_vector = tf.expand_dims(tf.convert_to_tensor([translation_vector[0], translation_vector[1], 0]), axis=0)

    
    def translator(self, x):
        expanded_translation = tf.tile(self.translation_vector, [tf.shape(x)[0], 1])
        return x + expanded_translation

    
class Rotation():
    def __init__(self, rotation_angle):
        self.rotation_angle = rotation_angle
    
    def rotation(self, x):
        cos_theta = tf.cos(self.rotation_angle)
        sin_theta = tf.sin(self.rotation_angle)
        rotation_matrix = tf.convert_to_tensor([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]], dtype=tf.float32)

        return  tf.matmul(x, tf.transpose(rotation_matrix))

class Scale():
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    
    def scale(self, x):
        return tf.multiply(x, self.scale_factor)



