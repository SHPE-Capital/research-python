"""SHPE Capital research system.

Layers, each behind a versioned contract (see docs/contracts.md):

    data        download, normalize, audit and snapshot market data
    features    point-in-time features computed from contract bars
    labels      outcomes that start strictly after the decision
    validation  chronological splits that keep the future out of training
    models      anything implementing the Model contract
    evaluation  reports comparable across every model
    experiments one config file -> one recorded run
"""

__version__ = "0.1.0"
