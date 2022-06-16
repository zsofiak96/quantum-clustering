import logging

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

logger = logging.getLogger(__name__)


def _map_function(x):
    """
    We map data feature values to theta and phi values using
    the following equation: phi = (x + 1) * (pi / 2)
    where phi is the phase and theta the angle
    :param x:  
    :return: 
    """
    return (x + 1) * np.pi / 2


def _map_features(input_point,
                  centroids,
                  n_centroids: int):
    """
    Map the input point and the centroids to theta and phi values
    via the _map_function method.
    :param input_point: Input point to map.
    :param centroids: Array of points to map.
    :param n_centroids: Number of centroids.
    :return: Tuple containing input point and centroids mapped.
    """
    phi_centroids_list = []
    theta_centroids_list = []
    phi_input = _map_function(input_point[0])
    theta_input = _map_function(input_point[1])
    for i in range(0, n_centroids):
        phi_centroids_list.append(_map_function(centroids[i][0]))
        theta_centroids_list.append(_map_function(centroids[i][1]))
    return phi_input, theta_input, phi_centroids_list, theta_centroids_list


def construct_circuit(input_point: np.ndarray,
                      centroids: np.ndarray,
                      k: int) -> QuantumCircuit:
    """
    Apply a Hadamard to the ancillary qubit and our mapped data points.
    Encode data points using U3 gate. Perform controlled swap to entangle
    the state with the ancillary qubit. Apply another Hadamard gate
    to the ancillary qubit.
                    ┌───┐                   ┌───┐
            |0anc>: ┤ H ├────────────■──────┤ H ├────────M
                    └───┘            |      └───┘
                    ┌───┐   ┌────┐   |
            |0>: ───┤ H ├───┤ U3 ├───X──────────
                    └───┘   └────┘   |
                    ┌───┐   ┌────┐   |
            |0>: ───┤ H ├───┤ U3 ├───X──────────
                    └───┘   └────┘
    :param input_point:  Input point from which calculate the distance.
    :param centroids: Array of points representing the centroids to calculate the distance to k.
    :param k: Number of centroids.
    :return: The quantum circuit created.
    """
    phi_input, theta_input, phi_centroids_list, theta_centroids_list = \
        _map_features(input_point, centroids, k)

    # Need 3 quantum registers, of size k one for a data point (input),
    # one for each centroid and one for each ancillary
    qreg_input = QuantumRegister(k, name='qreg_input')
    qreg_centroid = QuantumRegister(k, name='qreg_centroid')
    qreg_psi = QuantumRegister(k, name='qreg_psi')

    # Create a k bit ClassicalRegister to hold the result
    # of the measurements
    creg = ClassicalRegister(k, 'creg')

    # Create the quantum circuit containing our registers
    qc = QuantumCircuit(qreg_input, qreg_centroid, qreg_psi, creg, name='qc')

    for i in range(0, k):
        # Apply Hadamard
        qc.h(qreg_psi[i])
        qc.h(qreg_input[i])
        qc.h(qreg_centroid[i])

        # Encode new point and centroid
        qc.u(theta_input, phi_input, 0, qreg_input[i])
        qc.u(theta_centroids_list[i], phi_centroids_list[i], 0, qreg_centroid[i])

        # Perform controlled swap
        qc.cswap(qreg_psi[i], qreg_input[i], qreg_centroid[i])

        # Apply second Hadamard to ancillary
        qc.h(qreg_psi[i])

        # Measure ancillary
        qc.measure(qreg_psi[i], creg[i])

    return qc
