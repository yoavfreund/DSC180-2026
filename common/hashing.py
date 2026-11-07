import hashlib
import uuid


def new_id():
    return str(uuid.uuid4())


def sha256_array(array):
    import numpy as np
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()
