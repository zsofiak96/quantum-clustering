import logging
from typing import Dict, List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import Backend, BaseBackend
from qiskit.result import Result
from qiskit.tools import parallel_map
from qiskit.utils import QuantumInstance
from sklearn.base import ClusterMixin
from sklearn.exceptions import NotFittedError

from .circuits import construct_circuit
from .quantum_estimator import QuantumEstimator

logger = logging.getLogger(__name__)


class QFCMeans(ClusterMixin, QuantumEstimator):
    """
    A Quantum Fuzzy C-Means algorithm for classification
    """

    def __init__(
        self,
        n_clusters: int = 5,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
        *,
        init: Union[str, np.ndarray] = "random",
        n_init: int = 1,
        max_iter: int = 30,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        """
        :param n_clusters: The number of clusters to form as well as the number of centroids to generate.
        :param quantum_instance: the quantum instance to set. Can be a class qiskit.utils.QuantumInstance,
        a class qiskit.providers.Backend or a class qiskit.providers.BaseBackend.
        :param init: Method of initialization of centroids.
        :param n_init: Number of time the qfcmeans algorithm will be run with different centroid seeds.
        :param max_iter: Maximum number of iterations of the qfcmeans algorithm for a single run.
        :param tol: Tolerance with regard to the difference of the cluster centroids of two consecutive
        iterations to declare convergence.
        :param random_state: Determines random number generation for membership initialization.
        """
        super().__init__(quantum_instance=quantum_instance)
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_iter_ = 0
        self.tol = tol
        self.n_init = n_init
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_centers_ = None
        # do not rename : this name is needed for
        # `fit_predict` inherited method from
        # `ClusterMixin` base class
        self.labels_ = None        
    
    def _compute_distances_centroids(self, counts: Dict[str, int]) -> List[int]:
        """
        Compute distance, without explicitly measure it, of a point with respect
        to all the centroids using a dictionary of counts, which refers to the following circuit:
                        ┌───┐                   ┌───┐
                |0anc>: ┤ H ├────────────■──────┤ H ├────────M
                        └───┘            |      └───┘
                        ┌───┐   ┌────┐   |
                |0>: ───┤ H ├───┤ U3 ├───X──────────
                        └───┘   └────┘   |
                        ┌───┐   ┌────┐   |
                |0>: ───┤ H ├───┤ U3 ├───X──────────
                        └───┘   └────┘
        :param counts: Counts resulting after the simulation.
        :return: The computed distance.
        """
        distance_centroids = [0] * self.n_clusters
        x = 1
        for i in range(0, self.n_clusters):
            binary = format(x, "b").zfill(self.n_clusters)
            distance_centroids[i] = counts[binary] if binary in counts else 0
            x = x << 1
        return distance_centroids

    def _get_distances_centroids(self, results: Result) -> np.ndarray:
        """
        Retrieves distances from counts via :func:`_compute_distances_centroids`
        :param results: class qiskit.Result object of execution results
        :return: np.ndarray of distances
        """
        counts = results.get_counts()
        # compute distance from centroids using counts
        distances_list = list(
            map(lambda count: self._compute_distances_centroids(count), counts)
        )
        return np.asarray(distances_list)

    def _construct_circuits(self, X_test: np.ndarray) -> List[QuantumCircuit]:
        """
        Creates the circuits to be executed on
        the gated quantum computer for the classification
        process.
        :param X_test: The unclassified input data.
        :return: List of quantum circuits created for the computation.
        """
        logger.info("Starting circuits construction ...")
        """
        circuits = []
        for xt in X_test:
            circuits.append(construct_circuit(xt, self.cluster_centers_, self.n_clusters))
        """
        circuits = parallel_map(
            construct_circuit,
            X_test,
            task_args=[self.cluster_centers_, self.n_clusters],
        )

        logger.info("Done.")
        return circuits

    def _next_centers(X: np.ndarray, u: np.ndarray):
        """Update cluster centers."""
        um = u**2
        return (X.T @ um / np.sum(um, axis=0)).T
    
    def soft_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Soft predict of QFCMeans.
        :param X (np.ndarray): New data to predict.
        :return: Fuzzy partition array (np.ndarray) , returned as an array with n_samples rows and n_clusters columns.
        """
        circuits = self._construct_circuits(X)
        results = self.execute(circuits)
        temp = self._get_distances_centroids(results) ** (2 / (2 - 1))
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(
            temp.shape[-1], axis=1
        )
        np.nan_to_num(denominator_, nan=1)
        np.place(denominator_, denominator_<1, [1])
        denominator_ = temp[:, :, np.newaxis] / denominator_
        np.nan_to_num(denominator_, nan=1)
        np.place(denominator_, denominator_<1, [1])
        return 1 / denominator_.sum(2)

    def fit(self, X: np.ndarray):
        """
        Fits the model using X as training dataset. The fit model creates clusters
        from the training dataset given as input.
        :param X: training dataset
        :return: trained QFCMeans object
        """
        self.X_train = np.asarray(X)
        # initialize membership values U
        self.rng = np.random.default_rng(self.random_state)
        n_samples = self.X_train.shape[0]
        self.u = self.rng.uniform(size=(n_samples, self.n_clusters))
        self.u = self.u / np.tile(
            self.u.sum(axis=1)[np.newaxis].T, self.n_clusters
        )
        self.n_iter_ = 0
        error = np.inf

        while error > self.tol and self.n_iter_ < self.max_iter:
            u_old = self.u.copy()
            self.cluster_centers_ = QFCMeans._next_centers(self.X_train, self.u)
            self.u = self.soft_predict(self.X_train)
            for i in range(self.u.shape[0]):
                self.u[i] = (self.u[i] / self.u[i].sum())
            error = np.linalg.norm(self.u - u_old)
            self.n_iter_ = self.n_iter_ + 1
        self.labels_ = np.argmax(self.u, axis=1)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict the labels of the provided data.
        :param X_test: New data to predict.
        :return: Index of the cluster each sample belongs to.
        """
        if self.labels_ is None:
            raise NotFittedError(
                "This QFCMeans instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using "
                "this estimator."
            )

        return self.soft_predict(X_test).argmax(axis=-1)
