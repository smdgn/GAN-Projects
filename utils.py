<<<<<<< HEAD
import tensorflow as tf
import time
import os
import math

def cast_im(x):
    return tf.cast(tf.clip_by_value(x * 255, 0, 255), tf.uint8)

def get_session_name():
    current_time = int(time.time())
    description = input("Describe the session: ")
    session_name = "s{}: {}".format(current_time, description)
    print(session_name)
    return session_name

=======
import tensorflow as tf
import time
import os
import math

def cast_im(x):
    return tf.cast(tf.clip_by_value(x * 255, 0, 255), tf.uint8)

def get_session_name():
    current_time = int(time.time())
    description = input("Describe the session: ")
    session_name = "s{}: {}".format(current_time, description)
    print(session_name)
    return session_name

>>>>>>> cb74623bf62f8b9870e362de937ce970953c3eba
