"""QEC public API.

This package collects the main prior builders, decoders, and helper functions
used by the workflow QEC stages. It is the primary API surface for:

- selecting a decoder implementation
- building priors from syndrome data
- summarizing logical error outputs
"""

from musiq.qec.decoder import BPDecoder, MWPMDecoder, build_decoder_report, get_decoder, summarize_logical_error
from musiq.qec.interfaces import IDecoder, IPriorBuilder
from musiq.qec.mock import MockDecoder, MockPriorBuilder
from musiq.qec.prior import CirqPriorBuilder, StimPriorBuilder, build_prior_and_report

__all__ = [
    "IDecoder",
    "IPriorBuilder",
    "MockDecoder",
    "MockPriorBuilder",
    "MWPMDecoder",
    "BPDecoder",
    "get_decoder",
    "build_decoder_report",
    "StimPriorBuilder",
    "CirqPriorBuilder",
    "build_prior_and_report",
    "summarize_logical_error",
]
