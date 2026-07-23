"""Repo-local helpers for assembling *coupled* multi-field PyFVTool systems.

PyFVTool's :func:`~pyfvtool.solvePDE` / :func:`~pyfvtool.solveMatrixPDE` target a
**single field on a single mesh**.  Mechanistic models such as the General Rate
Model (:mod:`.grm`) instead couple *many* fields living on *several* meshes -- in
the GRM, ``nc`` bulk fields on an axial :class:`~pyfvtool.Grid1D` and
``nc * Nz`` particle fields on radial :class:`~pyfvtool.SphericalGrid1D` meshes,
linked by a film flux -- into one global sparse system.  Assembling that system
by hand means re-deriving global index maps (the old ``bidx`` / ``pidx``
closures), splicing sub-blocks by slice, and stamping the coupling fluxes with
hand-managed signs.

This module owns that bookkeeping so a model can declare *what* couples to *what*
and let the assembler place every block at the right global offset.

Scope / design notes
--------------------
* This is deliberately **outside** PyFVTool.  It builds on the public ``xxxTerm``
  operators and never touches PyFVTool internals.  See
  ``doc/pyfvtool_coupled_models.md`` for which pieces are candidates to upstream
  into PyFVTool itself (``CoupledSystem``, a ``couplingTerm``, and a conservative
  multi-component ``storageTerm``); until then they live here.
* The global solve is left to the caller (the GRM uses
  ``scipy.sparse.linalg.spsolve``): ``solveMatrixPDE`` is single-mesh and does
  not apply to a system spanning several meshes.
* Cell indices passed to :meth:`Assembler.couple` are **local flat cell numbers**
  in a field's raveled ``dims + 2`` block (ghost cells included).  For the 1D
  meshes used by the GRM this is just ``0 .. N+1`` with ``1 .. N`` the interior
  cells; for multi-D meshes use ``field.mesh.cell_numbers()`` to obtain them.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field

import numpy as np
import scipy.sparse as sp

__all__ = ["Field", "CoupledSystem", "Assembler", "conservative_storage"]


@dataclass(frozen=True)
class Field:
    """A single mesh-bound field occupying a contiguous slice of the global vector.

    Created by :meth:`CoupledSystem.add_field`; users do not construct it
    directly.  ``offset`` is the field's start index in the global unknown
    vector and ``size`` is its length *including* ghost cells
    (``prod(mesh.dims + 2)``).
    """

    mesh: object
    offset: int
    size: int
    name: str
    interior: np.ndarray = _dc_field(repr=False)  # global indices of interior cells

    @property
    def block(self) -> slice:
        """Slice of the global vector occupied by this field (incl. ghost cells)."""
        return slice(self.offset, self.offset + self.size)

    def gcell(self, local) -> np.ndarray | int:
        """Global index/indices of local flat cell number(s) ``local`` in this field."""
        return self.offset + np.asarray(local)


class CoupledSystem:
    """Layout manager for a global system of several mesh-bound fields.

    Assigns each registered field a contiguous slice of the global unknown
    vector and remembers the total size ``N``.  Hands out :class:`Assembler`
    instances that scatter per-field operator blocks and inter-field couplings
    into that global layout.

    Examples
    --------
    >>> sys = CoupledSystem()
    >>> a = sys.add_field(mz, "bulk")
    >>> b = sys.add_field(mp, "particle")
    >>> asm = sys.assembler()
    >>> asm.add_block(a, Mconv - Mdif_ax)
    >>> asm.couple(a, 3, b, Nr, coeff=k)        # film flux a(cell 3) <- b(cell Nr)
    >>> M, RHS = asm.matrix(), asm.rhs
    """

    def __init__(self) -> None:
        self._fields: list[Field] = []
        self.N: int = 0

    def add_field(self, mesh, name: str = "") -> Field:
        """Register ``mesh`` as a new field and return its :class:`Field` handle."""
        dims = np.asarray(mesh.dims)
        size = int(np.prod(dims + 2))
        G = mesh.cell_numbers()
        interior_local = G[tuple(slice(1, -1) for _ in dims)].ravel()
        f = Field(
            mesh=mesh,
            offset=self.N,
            size=size,
            name=name or f"field{len(self._fields)}",
            interior=self.N + interior_local,
        )
        self.N += size
        self._fields.append(f)
        return f

    @property
    def fields(self) -> list[Field]:
        return list(self._fields)

    def assembler(self) -> "Assembler":
        """Return a fresh :class:`Assembler` sized for the current layout."""
        return Assembler(self.N)


class Assembler:
    """Accumulates global matrix/RHS contributions, then emits a CSR system.

    Matrix entries are gathered as ``(row, col, value)`` triplets (COO) and built
    once in :meth:`matrix`, which is markedly cheaper than incremental
    ``lil_matrix`` splicing when many small blocks are stamped per step.
    """

    def __init__(self, N: int) -> None:
        self.N = N
        self._rows: list[np.ndarray] = []
        self._cols: list[np.ndarray] = []
        self._vals: list[np.ndarray] = []
        self.rhs = np.zeros(N)

    # -- matrix contributions ------------------------------------------------
    def add_block(self, field: Field, M_local) -> "Assembler":
        """Scatter a local ``(size x size)`` operator block to ``field``'s offset.

        ``M_local`` is any scipy sparse matrix sized for the field's mesh (e.g.
        the output of ``convectionUpwindTerm``, ``diffusionTerm``,
        ``boundaryConditionsTerm``), placed on the block diagonal at
        ``field.offset``.
        """
        coo = sp.coo_array(M_local)
        self._rows.append(coo.row + field.offset)
        self._cols.append(coo.col + field.offset)
        self._vals.append(coo.data)
        return self

    def add_entries(self, rows, cols, vals) -> "Assembler":
        """Add raw global ``(row, col, value)`` triplets (already global indices)."""
        self._rows.append(np.atleast_1d(np.asarray(rows)))
        self._cols.append(np.atleast_1d(np.asarray(cols)))
        self._vals.append(np.atleast_1d(np.asarray(vals, dtype=float)))
        return self

    def couple(self, field_a: Field, cell_a, field_b: Field, cell_b, coeff) -> "Assembler":
        """Stamp a linear inter-field flux ``coeff * (phi_a[cell_a] - phi_b[cell_b])``.

        Added to *field_a*'s equation rows: ``+coeff`` on the ``(a, a)`` diagonal
        and ``-coeff`` on the ``(a, b)`` off-diagonal.  ``cell_a`` / ``cell_b``
        are local flat cell numbers and may be arrays (vectorised over many
        couplings sharing the same form); ``coeff`` is broadcast against them.

        This is the discrete film / Robin transfer between two subdomains -- the
        building block that, in the GRM, links each bulk axial cell to its
        particle's outer shell.
        """
        ga = np.atleast_1d(field_a.gcell(cell_a))
        gb = np.atleast_1d(field_b.gcell(cell_b))
        coeff = np.broadcast_to(np.asarray(coeff, dtype=float), np.shape(ga))
        self.add_entries(np.concatenate([ga, ga]),
                         np.concatenate([ga, gb]),
                         np.concatenate([coeff, -coeff]))
        return self

    # -- RHS contributions ---------------------------------------------------
    def add_rhs(self, field: Field, vec) -> "Assembler":
        """Add a length-``size`` vector to ``field``'s RHS block."""
        self.rhs[field.block] += vec
        return self

    def add_rhs_at(self, gidx, vals) -> "Assembler":
        """Add ``vals`` to the global RHS at global indices ``gidx``."""
        np.add.at(self.rhs, gidx, vals)
        return self

    # -- emit ----------------------------------------------------------------
    def matrix(self) -> sp.csr_array:
        """Build and return the accumulated global matrix as a CSR array."""
        if not self._rows:
            return sp.csr_array((self.N, self.N))
        return sp.csr_array(
            (np.concatenate(self._vals),
             (np.concatenate(self._rows), np.concatenate(self._cols))),
            shape=(self.N, self.N),
        )


def conservative_storage(interior_gidx, capacity, cp_now, s_iter, s_old, dt):
    """Conservative multi-component accumulation (storage) term.

    Discretises ``d/dt[ storage_i(c) ] = ...`` for a set of competitively coupled
    fields whose stored amount ``storage_i`` depends on *all* component
    concentrations through a state-dependent capacity
    ``C_il = d storage_i / d c_l``.  Picard-linearised about the current iterate
    ``cp_now`` (``= c*``)::

        LHS:  sum_l C_il c_l / dt          (diagonal i==l plus competitive i!=l)
        RHS:  (sum_l C_il c*_l - s_iter_i + s_old_i) / dt

    At Picard convergence (``c == c*``) the linearisation cancels and the *exact*
    storage ``s_iter`` appears on the RHS, giving machine-precision mass
    conservation; ``s_old`` may be evaluated at an independent (old) auxiliary
    state -- e.g. the previous modulator -- so a moving salt gradient is captured
    exactly.  With a scalar diagonal ``C`` and ``s = alfa * c`` this reduces to
    PyFVTool's :func:`~pyfvtool.transientTerm`.

    Parameters
    ----------
    interior_gidx : array (nc, Ncell)
        Global indices of the interior cells, per component, in matching order.
    capacity : array (Ncell, nc, nc)
        Capacity Jacobian ``C[:, i, l] = d storage_i / d c_l`` per cell.
    cp_now, s_iter, s_old : array (nc, Ncell)
        Current iterate concentrations, storage at ``cp_now``, and storage at the
        previous time level (with the old auxiliary state).
    dt : float
        Time step.

    Returns
    -------
    rows, cols, vals : np.ndarray
        Global matrix triplets for the LHS capacity blocks.
    rhs_idx, rhs_vals : np.ndarray
        Global RHS indices and values.
    """
    interior_gidx = np.asarray(interior_gidx)
    nc, ncell = interior_gidx.shape
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    rhs_idx: list[np.ndarray] = []
    rhs_vals: list[np.ndarray] = []
    for i in range(nc):
        ri = interior_gidx[i]
        lin_inc = np.zeros(ncell)                       # sum_l C_il c*_l
        for l in range(nc):
            coef = capacity[:, i, l] / dt
            rows.append(ri)
            cols.append(interior_gidx[l])
            vals.append(coef)
            lin_inc += capacity[:, i, l] * cp_now[l]
        rhs_idx.append(ri)
        rhs_vals.append((lin_inc - s_iter[i] + s_old[i]) / dt)
    return (np.concatenate(rows), np.concatenate(cols), np.concatenate(vals),
            np.concatenate(rhs_idx), np.concatenate(rhs_vals))
