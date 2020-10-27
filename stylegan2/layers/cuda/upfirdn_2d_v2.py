import os
import numpy as np
import tensorflow as tf
from stylegan2.layers.cuda import custom_ops


def _get_plugin():
    return custom_ops.get_plugin(os.path.splitext(__file__)[0] + '.cu')

def upfirdn_2d(x, k, upx=1, upy=1, downx=1, downy=1, padx0=0, padx1=0, pady0=0, pady1=0, impl='cuda'):

    impl_dict = {
        'ref':  _upfirdn_2d_ref,
        'cuda': _upfirdn_2d_ref
    }
    return impl_dict[impl](x=x, k=k, upx=upx, upy=upy, downx=downx, downy=downy, padx0=padx0, padx1=padx1, pady0=pady0, pady1=pady1)


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def compute_paddings(resample_kernel, convW, up, down, is_conv, factor=2, gain=1):
    assert not (up and down)

    k = [1] * factor if resample_kernel is None else resample_kernel
    if up:
        k = _setup_kernel(k) * (gain * (factor ** 2))
        if is_conv:
            p = (k.shape[0] - factor) - (convW - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
        else:
            p = k.shape[0] - factor
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2
    elif down:
        k = _setup_kernel(k) * gain
        if is_conv:
            p = (k.shape[0] - factor) + (convW - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
        else:
            p = k.shape[0] - factor
            pad0 = (p + 1) // 2
            pad1 = p // 2
    else:
        k = resample_kernel
        pad0, pad1 = 0, 0
    return k, pad0, pad1


def upsample_2d(x, x_res, pad0, pad1, k, factor=2):
    assert isinstance(factor, int) and factor >= 1
    return _simple_upfirdn_2d(x, x_res, k, up=factor, pad0=pad0, pad1=pad1)


def downsample_2d(x, x_res, pad0, pad1, k, factor=2):
    assert isinstance(factor, int) and factor >= 1
    return _simple_upfirdn_2d(x, x_res, k, down=factor, pad0=pad0, pad1=pad1)

def _shape(tf_expr, dim_idx):
    if tf_expr.shape.rank is not None:
        dim = tf_expr.shape[dim_idx].value
        if dim is not None:
            return dim
    return tf.shape(tf_expr)[dim_idx]


def upsample_conv_2d(x, x_res, w, convH, convW, pad0, pad1, k, factor=2):
    assert isinstance(factor, int) and factor >= 1

    # Check weight shape.
    w = tf.convert_to_tensor(w)
    assert w.shape.rank == 4
    # convH = w.shape[0]
    # convW = w.shape[1]
    inC = tf.shape(w)[2]
    outC = tf.shape(w)[3]
    assert convW == convH

    # Determine data dimensions.
    stride = [1, 1, factor, factor]
    output_shape = [tf.shape(x)[0], outC, (x_res - 1)
                    * factor + convH, (x_res - 1) * factor + convW]
    num_groups = tf.shape(x)[1] // inC

    # Transpose weights.
    w = tf.reshape(w, [convH, convW, inC, num_groups, -1])
    w = tf.transpose(w[::-1, ::-1], [0, 1, 4, 3, 2])
    w = tf.reshape(w, [convH, convW, -1, num_groups * inC])

    # Execute.
    x = tf.nn.conv2d_transpose(x, w, output_shape=output_shape,
                               strides=stride, padding='VALID', data_format='NCHW')
    new_x_res = output_shape[2]
    return _simple_upfirdn_2d(x, new_x_res, k, pad0=pad0, pad1=pad1)


def conv_downsample_2d(x, x_res, w, convH, convW, pad0, pad1, k, factor=2):
    assert isinstance(factor, int) and factor >= 1
    w = tf.convert_to_tensor(w)
    # convH, convW, _inC, _outC = w.shape.as_list()
    assert convW == convH

    s = [1, 1, factor, factor]
    x = _simple_upfirdn_2d(x, x_res, k, pad0=pad0, pad1=pad1)
    return tf.nn.conv2d(x, w, strides=s, padding='VALID', data_format='NCHW')


def _simple_upfirdn_2d(x, x_res, k, up=1, down=1, pad0=0, pad1=0):
    assert x.shape.rank == 4
    y = x
    y = tf.reshape(y, [-1, x_res, x_res, 1])
    y = upfirdn_2d(y, k, upx=up, upy=up, downx=down,
                        downy=down, padx0=pad0, padx1=pad1, pady0=pad0, pady1=pad1)
    y = tf.reshape(y, [-1, tf.shape(x)[1], tf.shape(y)[1], tf.shape(y)[2]])
    return y


def _upfirdn_2d_ref(x, k, upx, upy, downx, downy, padx0, padx1, pady0, pady1):
    """Slow reference implementation of `upfirdn_2d()` using standard TensorFlow ops."""

    x = tf.convert_to_tensor(x)
    k = np.asarray(k, dtype=np.float32)
    assert x.shape.rank == 4
    inH = x.shape[1].value
    inW = x.shape[2].value
    minorDim = _shape(x, 3)
    kernelH, kernelW = k.shape
    assert inW >= 1 and inH >= 1
    assert kernelW >= 1 and kernelH >= 1
    assert isinstance(upx, int) and isinstance(upy, int)
    assert isinstance(downx, int) and isinstance(downy, int)
    assert isinstance(padx0, int) and isinstance(padx1, int)
    assert isinstance(pady0, int) and isinstance(pady1, int)

    # Upsample (insert zeros).
    x = tf.reshape(x, [-1, inH, 1, inW, 1, minorDim])
    x = tf.pad(x, [[0, 0], [0, 0], [0, upy - 1], [0, 0], [0, upx - 1], [0, 0]])
    x = tf.reshape(x, [-1, inH * upy, inW * upx, minorDim])

    # Pad (crop if negative).
    x = tf.pad(x, [[0, 0], [max(pady0, 0), max(pady1, 0)],
                   [max(padx0, 0), max(padx1, 0)], [0, 0]])
    x = x[:, max(-pady0, 0): x.shape[1].value - max(-pady1, 0),
          max(-padx0, 0): x.shape[2].value - max(-padx1, 0), :]

    # Convolve with filter.
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, [-1, 1, inH * upy + pady0 +
                       pady1, inW * upx + padx0 + padx1])
    w = tf.constant(k[::-1, ::-1, np.newaxis, np.newaxis], dtype=x.dtype)
    x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1],
                     padding='VALID', data_format='NCHW')
    x = tf.reshape(x, [-1, minorDim, inH * upy + pady0 + pady1 -
                       kernelH + 1, inW * upx + padx0 + padx1 - kernelW + 1])
    x = tf.transpose(x, [0, 2, 3, 1])

    # Downsample (throw away pixels).
    return x[:, ::downy, ::downx, :]
