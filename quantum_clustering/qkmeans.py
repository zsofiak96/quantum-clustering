import logging
import warnings
from copy import deepcopy
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


class QKMeans(ClusterMixin, QuantumEstimator):
    """
    The Quantum K-Means algorithm for classification
    Note:
        The naming conventions follow the KMeans from
        sklearn.cluster
    """

    def __init__(
        self,
        n_clusters: int = 5,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
        *,
        max_iter: int = 30,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        """
        Args:
            n_clusters:
                The number of clusters to form as well as the number of
                centroids to generate.
            quantum_instance:
                the quantum instance to set. Can be a
                :class:`~qiskit.utils.QuantumInstance`, a :class:`~qiskit.providers.Backend`
                or a :class:`~qiskit.providers.BaseBackend`
            max_iter:
                Maximum number of iterations of the qkmeans algorithm for a
                single run.
            tol:
                Tolerance with regard to the difference of the cluster centroids
                of two consecutive iterations to declare convergence.
            random_state:
                Determines random number generation for centroid initialization.
        """
        super().__init__(quantum_instance=quantum_instance)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_iter_ = 0
        self.tol = tol
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def _init_centroid(
        self, X: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
        """
        Create random cluster centroids.
        Args:
            X:
                The dataset to be used for centroid initialization.
            n_clusters:
                The desired number of clusters for which centroids are required.
            random_state:
                Determines random number generation for centroid initialization.
        Returns:
            Collection of k centroids as a numpy ndarray.
        """
        np.random.seed(random_state)
        centroids = []
        m = np.shape(X)[0]

        for _ in range(n_clusters):
            r = np.random.randint(0, m - 1)
            centroids.append(X[r])

        self.cluster_centers_ = np.array(centroids)
        return self.cluster_centers_

    def _recompute_centroids(self):
        """
        Reassign centroid value to be the calculated mean value for each cluster.
        If a cluster is empty the corresponding centroid remains the same.
        """
        for i in range(self.n_clusters):
            if np.sum(self.labels_ == i) != 0:
                self.cluster_centers_[i] = np.mean(
                    self.X_train[self.labels_ == i], axis=0
                )

    def _compute_distances_centroids(self, counts: Dict[str, int]) -> List[int]:
        """
        Compute distance, without explicitly measure it, of a point with respect
        to all the centroids using a dictionary of counts,
        which refers to the following circuit:
        .. parsed-literal::
                        ┌───┐                   ┌───┐
                |0anc>: ┤ H ├────────────■──────┤ H ├────────M
                        └───┘            |      └───┘
                        ┌───┐   ┌────┐   |
                |0>: ───┤ H ├───┤ U3 ├───X──────────
                        └───┘   └────┘   |
                        ┌───┐   ┌────┐   |
                |0>: ───┤ H ├───┤ U3 ├───X──────────
                        └───┘   └────┘
        Args:
            counts:
                Counts resulting after the simulation.
        Returns:
            The computed distance.
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
        Args:
            results: :class:`~qiskit.Result` object of execution results
        Returns:
            np.ndarray of distances
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
        process
        Args:
            X_test: The unclassified input data.
        Returns:
            List of quantum circuits created for the computation
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

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fits the model using X as training dataset
        and y as training labels. For the qkmeans algorithm y is ignored.
        The fit model creates clusters from the training dataset given as input
        Args:
            X: training dataset
            y: Ignored.
               Kept here for API consistency
        Returns:
            trained QKMeans object
        """
        self.X_train = np.asarray(X)
        self._init_centroid(self.X_train, self.n_clusters, self.random_state)
        self.labels_ = np.zeros(self.X_train.shape[0])
        error = np.inf
        self.n_iter_ = 0

        # while error not below tolerance, reiterate the
        # centroid computation for a maximum of `max_iter` times
        while error > self.tol and self.n_iter_ < self.max_iter:
            # construct circuits using training data
            # notice: the construction uses the centroids
            # which are recomputed after every iteration
            circuits = self._construct_circuits(self.X_train)

            # executing and computing distances from centroids
            results = self.execute(circuits)
            distances = self._get_distances_centroids(results)

            # assigning clusters and recomputing centroids
            self.labels_ = np.argmin(distances, axis=1)
            cluster_centers_old = deepcopy(self.cluster_centers_)
            self._recompute_centroids()

            # evaluating error and updating iteration count
            error = np.linalg.norm(self.cluster_centers_ - cluster_centers_old)
            self.n_iter_ = self.n_iter_ + 1

        if self.n_iter_ == self.max_iter:
            warnings.warn(
                f"QKMeans failed to converge after " f"{self.max_iter} iterations."
            )

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict the labels of the provided data.
        Args:
            X_test:
                New data to predict.
        Returns:
            Index of the cluster each sample belongs to.
        """
        if self.labels_ is None:
            raise NotFittedError(
                "This QKMeans instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using "
                "this estimator."
            )

        circuits = self._construct_circuits(X_test)
        results = self.execute(circuits)
        distances = self._get_distances_centroids(results)

        predicted_labels = np.argmin(distances, axis=1)
        return predicted_labels
