from musiq.analysis.common.metrics_utils import _basis_labels
from musiq.analysis.common.state_utils import basis_labels


def test_basis_labels_follow_reference_bitstring_order():
    labels = basis_labels(dimension=16, num_qubits=4, levels=2)

    assert labels[10] == "0101"
    assert _basis_labels(dimension=16, num_qubits=4, levels=2)[10] == "0101"

