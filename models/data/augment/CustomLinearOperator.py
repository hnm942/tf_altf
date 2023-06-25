import tensorflow as tf

class LinearOperatorTranslation(tf.linalg.LinearOperator):
    def __init__(self, translation_vector):
        super(LinearOperatorTranslation, self).__init__(dtype= tf.float32, is_non_singular= True, is_self_adjoint= True, is_positive_definite= True)
        self.translation_vector = translation_vector # shape: (2, 1)
    
    def _shape(self):
        return tf.TensorShape([2, 2])
    
    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        translation_matrix = tf.linalg.LinearOperatorFullMatrix(tf.expand_dims(self.translation_vector, axis = 0)) # (2, ) --> (1, 2)
        return tf.matmul(x, translation_matrix.to_dense(), adjoint_b= adjoint)
    
class LinearOperatorRotation(tf.linalg.LinearOperator):
    def __init__(self, rotation_angle):
        super(LinearOperatorRotation, self).__init__(dtype=tf.float32, is_non_singular= True, is_self_adjoint=True, is_positive_definite= True)
        self.rotation_angle = rotation_angle

    def _shape(self):
        return tf.TensorShape([2, 2])
    
    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        cos_theta = tf.cos(self.rotation_angle)
        sin_theta = tf.sin(self.rotation_angle)
        rotation_matrix = tf.convert_to_tensor([[cos_theta, -sin_theta],[sin_theta, cos_theta]], dtype= self.dtype)
        return tf.matmul(x, rotation_matrix, adjoint_b=adjoint)

class LinearOperatorScale(tf.linalg.LinearOperator):
    def __init__(self, scale_factor):
        super(LinearOperatorScale, self).__init__(dtype= tf.float32, is_non_singular= True, is_positive_definite=True)
        self.scale_factor = scale_factor

    def _shape(self):
        return tf.TensorShape([2, 2])
    
    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        scale_matrix  = tf.linalg.LinearOperatorScaledIdentity(tf.eye(2), self.scale_factor)
        return tf.matmul(x, scale_matrix.to_dense(), adjoint_b = adjoint)


