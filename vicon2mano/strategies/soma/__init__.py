"""strategySOMA: learned, label-free per-frame marker labelling.

Hand adaptation of SOMA (Ghorbani & Black, ICCV 2021): a self-attention
network over the raw, unordered marker cloud, trained only on synthetic
clouds generated from a MANO surface model of where the markers really sit.

This strategy is self-contained: it imports from ``vicon2mano.core`` only,
never from another strategy or from ``scripts/``.
"""
