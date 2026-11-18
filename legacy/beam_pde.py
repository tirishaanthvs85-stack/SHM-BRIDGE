"""
==========================================================
Bridge SHM using Physics-Informed Neural Networks (PINN)
beam_pde.py
----------------------------------------------------------
Physics Module

Euler-Bernoulli Beam Equation

EI * d4w/dx4 = q

==========================================================
"""

import tensorflow as tf
from config import *


class BeamPDE:

    def __init__(self):

        self.E = tf.constant(E, dtype=tf.float32)
        self.I = tf.constant(I, dtype=tf.float32)
        self.q = tf.constant(Q, dtype=tf.float32)
        self.L = tf.constant(L, dtype=tf.float32)

    # ----------------------------------------------------
    # First, Second, Third and Fourth Derivatives
    # ----------------------------------------------------

    def derivatives(self, model, x):

        """
        Returns

        w
        dw/dx
        d2w/dx2
        d3w/dx3
        d4w/dx4
        """

        x = tf.convert_to_tensor(x, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape4:

            tape4.watch(x)

            with tf.GradientTape(persistent=True) as tape3:

                tape3.watch(x)

                with tf.GradientTape(persistent=True) as tape2:

                    tape2.watch(x)

                    with tf.GradientTape() as tape1:

                        tape1.watch(x)

                        w = model(x)

                    dw = tape1.gradient(w, x)

                d2w = tape2.gradient(dw, x)

            d3w = tape3.gradient(d2w, x)

        d4w = tape4.gradient(d3w, x)

        del tape4
        del tape3
        del tape2

        return w, dw, d2w, d3w, d4w

    # ----------------------------------------------------
    # Physics Residual
    # ----------------------------------------------------

    def residual(self, model, x):

        """
        Residual:

        EI*w'''' - q = 0
        """

        w, dw, d2w, d3w, d4w = self.derivatives(model, x)

        residual = self.E * self.I * d4w - self.q

        return residual

    # ----------------------------------------------------
    # Boundary Conditions
    # ----------------------------------------------------

    def boundary_residual(self, model):

        """
        Simply Supported Beam

        w(0)=0
        w(L)=0

        w''(0)=0
        w''(L)=0
        """

        x0 = tf.constant([[0.0]], dtype=tf.float32)
        xL = tf.constant([[L]], dtype=tf.float32)

        w0, _, d2w0, _, _ = self.derivatives(model, x0)

        wL, _, d2wL, _, _ = self.derivatives(model, xL)

        bc = {

            "w0": w0,

            "wL": wL,

            "m0": d2w0,

            "mL": d2wL

        }

        return bc

    # ----------------------------------------------------
    # PDE Loss
    # ----------------------------------------------------

    def physics_loss(self, model, x):

        r = self.residual(model, x)

        return tf.reduce_mean(tf.square(r))

    # ----------------------------------------------------
    # Boundary Loss
    # ----------------------------------------------------

    def boundary_loss(self, model):

        bc = self.boundary_residual(model)

        loss = (

            tf.reduce_mean(tf.square(bc["w0"])) +

            tf.reduce_mean(tf.square(bc["wL"])) +

            tf.reduce_mean(tf.square(bc["m0"])) +

            tf.reduce_mean(tf.square(bc["mL"]))

        )

        return loss

    # ----------------------------------------------------
    # Complete Physics
    # ----------------------------------------------------

    def total_physics_loss(self, model, x):

        lpde = self.physics_loss(model, x)

        lbc = self.boundary_loss(model)

        return lpde + lbc