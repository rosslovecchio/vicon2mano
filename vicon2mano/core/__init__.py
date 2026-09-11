"""Strategy-agnostic building blocks.

Everything here is shared by more than one relabelling strategy. It lives
in the package (not in ``scripts/``) because strategies import it as a
library; previously this plumbing sat inside
``scripts/relabel_with_mano.py``, which meant every other strategy had to
import the MANO driver just to locate a trial CSV.
"""
