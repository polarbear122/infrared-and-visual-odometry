import tensorflow as tf
import numpy as np


def Strategy(content, style):
    # return tf.reduce_sum(content, style)
    print('content:{}'.format(content))
    print('style:{}'.format(style))
    concat = tf.concat([content,style],3)
    # return content+style
    print('concat:{}'.format(concat))
    # return content+style
    return concat



