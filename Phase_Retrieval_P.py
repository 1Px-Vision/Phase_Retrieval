#Function: Physical Model GS, Fourier_Born, Fourier_Rytov: Amplitude-based algorithms for phase retrieval

import numpy as np
import numpy
from PIL import Image
import matplotlib.pyplot as plt
import os
import math
import tensorflow as tf
import cv2
from tensorflow import keras
import math
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from math import log10, sqrt

#Function Algorithm 1
def fftshift(x, axes=None):
    axes = tuple(range(2))
    shift = [dim // 2 for dim in [int(x.shape[0]), int(x.shape[1])]]
    result = tf.roll(x, shift, axis=axes)
    result = tf.cast(result, tf.complex64)
    return result

def ifftshift(x):
    axes = tuple(range(2))
    shift = [-(dim // 2) for dim in [int(x.shape[0]), int(x.shape[1])]]
    result = tf.roll(x, shift, axis=axes)
    result = tf.cast(result, tf.complex64)
    return result

def GS(inpt, lamb, L, Z):
    M = int(inpt.shape[1])
    image = tf.cast(inpt, tf.complex128)
    image = 1j * image
    U_in = tf.exp(image)
    U_out = ifftshift(tf.signal.fft2d(fftshift(U_in)))
    fx = 1/L
    x = tf.linspace(-M/2, M/2-1, M)
    fx = fx * x
    [Fx, Fy] = tf.meshgrid(fx, fx)
    k = 2 * math.pi / lamb
    H = tf.sqrt(1 - lamb * lamb * (Fx * Fx + Fy * Fy))
    temp = k * Z * H
    temp = tf.cast(temp, tf.complex64)
    H = tf.exp(1j * temp)
    U_out = U_out * H
    U_out = ifftshift(tf.signal.ifft2d(fftshift(U_out)))
    I1 = tf.abs(U_out) * tf.abs(U_out)
    I1 = I1 / tf.reduce_max(tf.reduce_max(I1))
    return I1

#Function Algorithm 2
def phase_retrieval_fourier_Born(object_intensity, lamb, L, Z):
    # Define the spatial frequency grid
    N = object_intensity.shape[0]
    df = 1.0 / (L * N)
    f = tf.linspace(-N / 2, N / 2 - 1, N) * df

    # Construct the transfer function
    H = tf.exp(-1j * 2 * np.pi * Z / lamb * tf.sqrt(1 - tf.cast((lamb * f) ** 2, tf.complex64)))

    object_intensity = tf.cast(object_intensity, tf.complex64)

    # Perform the Fourier transform of the input intensity
    F_object_intensity = tf.signal.fftshift(tf.signal.fft2d(object_intensity))

    # Apply the transfer function in Fourier domain
    real_part = tf.math.real(F_object_intensity)
    imag_part = tf.math.imag(F_object_intensity)
    F_retrieved_intensity = tf.complex(real_part * tf.math.real(H) - imag_part * tf.math.imag(H),
                                       real_part * tf.math.imag(H) + imag_part * tf.math.real(H))

    # Retrieve the phase by inverse Fourier transforming the modified intensity
    retrieved_phase = tf.math.angle(tf.signal.ifft2d(tf.signal.ifftshift(F_retrieved_intensity)))

    return retrieved_phase

#Function Algorithm 3
def phase_retrieval_fourier_Rytov(object_intensity, lamb, L, Z):
    # Create the spatial grid
    x = tf.linspace(-L/2, L/2, tf.shape(object_intensity)[0])
    y = tf.linspace(-L/2, L/2, tf.shape(object_intensity)[1])
    X, Y = tf.meshgrid(x, y)

    # Compute the Fourier transform of the object intensity
    object_ft = tf.signal.fft2d(tf.cast(tf.sqrt(object_intensity), tf.complex64))

    # Compute the propagation kernel in the Fourier domain
    k = 2 * np.pi / lamb
    fx = tf.linspace(-1/(2*L), 1/(2*L), tf.shape(object_intensity)[0])
    fy = tf.linspace(-1/(2*L), 1/(2*L), tf.shape(object_intensity)[1])
    FX, FY = tf.meshgrid(fx, fy)
    H = tf.exp(1j * k * Z * tf.cast(tf.sqrt(1 - lamb**2 * (FX**2 + FY**2)), tf.complex64))

    # Perform the phase retrieval iteration
    num_iterations = 100
    for i in range(num_iterations):
        # Estimate the phase of the object
        phase = tf.math.angle(tf.signal.ifft2d(object_ft))

        object_intensity = tf.cast(object_intensity, tf.float32)
        object_sqrt = tf.sqrt(object_intensity)
        object_complex = tf.complex(object_sqrt, tf.zeros_like(object_sqrt))

        object_ft = tf.multiply(object_complex, tf.exp(1j * tf.cast(phase, tf.complex64)))

        # Propagate the updated object in the Fourier domain
        propagated_ft = object_ft * H

        # Compute the inverse Fourier transform to get the updated object
        object_intensity = tf.square(tf.abs(tf.signal.ifft2d(propagated_ft)))

    return object_intensity
