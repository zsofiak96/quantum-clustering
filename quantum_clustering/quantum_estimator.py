import logging
from typing import List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend, BaseBackend
from qiskit.result import Result
from qiskit.utils import QuantumInstance
from sklearn.base import TransformerMixin

logger = logging.getLogger(__name__)


class QuantumEstimator(TransformerMixin):
    def __init__(
        self,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
    ):
        """
        :param quantum_instance: The quantum instance to set. Can be a
        class qiskit.utils.QuantumInstance, a class qiskit.providers.Backend
        or a class qiskit.providers.BaseBackend
        """
        self.X_train = np.asarray([])
        self.y_train = np.asarray([])
        self._set_quantum_instance(quantum_instance)

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Returns the quantum instance to evaluate the circuit."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(
        self, quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]]
    ):
        """Quantum Instance setter"""
        self._set_quantum_instance(quantum_instance)

    def _set_quantum_instance(
        self, quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]]
    ):
        """
        Internal method to set a quantum instance according to its type
        :param quantum_instance: The quantum instance to set. Can be a
        class qiskit.utils.QuantumInstance, a class qiskit.providers.Backend
        or a class qiskit.providers.BaseBackend
        """
        if isinstance(quantum_instance, (BaseBackend, Backend)):
            quantum_instance = QuantumInstance(quantum_instance)

        self._quantum_instance = quantum_instance

    def execute(
        self, qcircuits: Union[QuantumCircuit, List[QuantumCircuit]]
    ) -> Union[Optional[Result], None]:
        """
        Executes the given quantum circuit.
        :param qcircuits: a class qiskit.QuantumCircuit or a list of this type to be executed.
        :return: the execution results.
        """
        if self._quantum_instance is None:
            raise QiskitError("Circuits execution requires a quantum instance")

        logger.info("Executing circuits...")

        # Instead of transpiling and assembling the quantum object
        # and running the backend, call execute from the quantum
        # instance that does it at once a very efficient way
        # notice: this execution is parallelized
        # which is why a list of circuits is passed and not one at a time
        result = self._quantum_instance.execute(qcircuits)
        return result
