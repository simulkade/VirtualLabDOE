# Coupled PyFVTool models: making the bookkeeping easier

*Investigation + implementation notes — 2026-06-16*

This note reviews `src/vlab_doe/models/chromatography/grm.py` (the
finite-volume General Rate Model) against the PyFVTool API and records two
things:

1. **What is now implemented in this package** — a repo-local assembly layer
   (`src/vlab_doe/models/chromatography/coupling.py`) that removes the hand
   bookkeeping from the GRM, plus the tier-1 use of PyFVTool's own
   `transientTerm`.
2. **What is deliberately deferred** — a separate list of pieces that *could*
   move into PyFVTool itself later, kept apart so that decision stays open.

The split is intentional: everything in §2 lives in this repo today; nothing in
§4 has been added to PyFVTool.

---

## 1. The problem the GRM exposed

The GRM couples many fields on several meshes: `nc` bulk fields on an axial
`Grid1D` and `nc * Nz` particle fields on radial `SphericalGrid1D` meshes, linked
by a film flux, assembled into **one** global sparse system. PyFVTool's
`solvePDE` / `solveMatrixPDE` target a *single field on a single mesh*, so the
original solver did the coupling by hand:

- a global `scipy.sparse.lil_matrix` with two index closures `bidx(i,a)` /
  `pidx(i,j,k)` and a manual total size `N`;
- sub-blocks spliced in by slice (`M[b0:b0+nb, b0:b0+nb] += …`);
- the transient term built as a hand-padded `sp.diags`;
- the film flux stamped as four sign-managed entries per `(i,j)`;
- the competitive storage balance scattered cell-by-cell.

All correct (mass balance at ~1e-14), but only the author could safely touch it,
and every future multi-domain model would re-derive the same index algebra.

---

## 2. Implemented in this package

### 2.1 `coupling.py` — a `CoupledSystem` assembler (repo-local)

`coupling.py` owns the global layout so models declare *what couples to what*
instead of computing offsets:

- **`CoupledSystem.add_field(mesh, name)` → `Field`** assigns each field a
  contiguous global slice and records its interior global indices. `bidx`,
  `pidx`, `part_offset`, and the manual `N` are gone — `N = sys.N`.
- **`Assembler.add_block(field, M_local)`** scatters any per-mesh operator block
  (`convectionUpwindTerm`, `diffusionTerm`, `boundaryConditionsTerm`,
  `transientTerm`) to the field's offset via COO triplets — no `lil_matrix`
  splicing.
- **`Assembler.couple(field_a, cell_a, field_b, cell_b, coeff)`** stamps the
  linear inter-field flux `coeff*(phi_a - phi_b)` (`+coeff` on `(a,a)`, `-coeff`
  on `(a,b)`) in one signed, tested place. The GRM's whole film section is now
  two `couple` calls per `(i,j)`.
- **`Assembler.matrix()`** builds the global CSR once from the accumulated
  triplets (cheaper than incremental `lil`).

Because each particle `(i,j)` is registered as its own small spherical field, the
diffusion/BC blocks stay naturally block-diagonal and the film flux is a per-cell
coupling rather than an index computation.

### 2.2 `conservative_storage` — the competitive accumulation term (repo-local)

The GRM's most subtle piece — the conservative, competitively-coupled,
modulator-aware storage balance — is lifted verbatim into
`coupling.conservative_storage(...)`. It returns global matrix triplets and RHS
entries for

```
LHS:  sum_l C_il c_l / dt
RHS:  (sum_l C_il c*_l - s_iter_i + s_old_i) / dt
```

so that at Picard convergence the *exact* storage appears (machine-precision mass
conservation) and `s_old`, evaluated at the **old** modulator, captures a moving
salt gradient exactly. This is the part genuinely novel relative to PyFVTool's
toolbox; keeping it as a named function documents it and makes it reusable by any
adsorption/ion-exchange/reactive-transport model in this repo.

### 2.3 `transientTerm` for the accumulation terms

The modulator advance and the bulk accumulation now call PyFVTool's
`transientTerm(phi_old, dt, alfa)` instead of hand-built `sp.diags` + RHS. With
`alfa = 1` it returns exactly the `1/dt` interior diagonal and the `c_old/dt`
interior RHS the code used to assemble, and it restricts to interior cells for
us (no more `np.concatenate([[0.0], …, [0.0]])` ghost padding).

### 2.4 Result

`run_grm` is unchanged in physics and output: all GRM tests pass and the
multi-component gradient mass-balance error is `-1.9e-14` (vs the 1e-8 test bar).
The per-step kernel is now declarative — register fields, add blocks, `couple`,
`conservative_storage`, solve — with the index algebra owned by `coupling.py`.

### 2.5 `gradientTerm` for the inlet dispersive flux

The inlet boundary flux used for the mass balance is a total flux
`u*c_face - Dax*dc/dz` at the inlet face. The dispersive part was a hand finite
difference `Dax*(feed-c1)/(dz/2)`; it is now `-Dax * gradientTerm(c)._xvalue[0]`.
Because the global solve includes the ghost-cell BC rows, the solution vector
already carries the Dirichlet-consistent inlet ghost value (`2*feed - c1`), so
`gradientTerm` reads the inlet-face gradient straight off the solved field with no
BC re-application, and is correct on non-uniform meshes (the half-spacing is
computed by the operator). Verified algebraically identical: the multi-component
gradient mass-balance error is unchanged to the bit (`-1.906e-14`).

### 2.6 What was *not* changed, on purpose

- **The solve stays `scipy.sparse.linalg.spsolve`.** `solveMatrixPDE` is a
  single-mesh helper (it reshapes the solution back onto one mesh, pdesolver.py:147)
  and does not apply to a system spanning the bulk and particle meshes. `spsolve`
  on the assembled global `M`/`RHS` is the correct call here.
- **`upwindMean` left latent.** `upwindMean(C, u_face)` is the right tool for an
  iterated *nonlinear* convection term (velocity depending on state); the GRM has
  constant `u`, so it is noted here as the hook to use the moment velocity stops
  being constant, but not wired in.

---

## 3. Sketch of the refactored inner loop

```python
sys  = CoupledSystem()
bulk = [sys.add_field(mz, f"c{i}") for i in range(nc)]
part = [[sys.add_field(mp, f"cp{i}_{j}") for j in range(Nz)] for i in range(nc)]

# static geometry: bulk transport, particle diffusion+BC, film couplings
asm0 = sys.assembler()
for i in range(nc):
    asm0.add_block(bulk[i], Mconv - Mdif_ax)
    for j in range(Nz):
        asm0.add_block(part[i][j], -Mdif_p[i] + Mbc_p)
        asm0.couple(bulk[i], 1 + j, part[i][j], Nr, kb[i])
        asm0.couple(part[i][j], Nr, bulk[i], 1 + j, kp[i])
M_static = asm0.matrix()

# per Picard sweep
asm = sys.assembler()
for i in range(nc):
    asm.add_block(bulk[i], Mbc_b).add_rhs(bulk[i], RHSbc_b)
    asm.add_block(bulk[i], *pf.transientTerm(c_old_cv[i], dt, 1.0)...)   # accumulation
rows, cols, vals, ridx, rvals = conservative_storage(part_interior, C, cp_now, s_iter, s_old, dt)
asm.add_entries(rows, cols, vals).add_rhs_at(ridx, rvals)
x = spla.spsolve((M_static + asm.matrix()).tocsr(), asm.rhs)
```

---

## 4. Deferred: candidates to upstream into PyFVTool (NOT done)

These are kept separate so the decision to move them into the library stays open.
`coupling.py` is written to depend only on PyFVTool's public `xxxTerm` operators,
so any of these could be promoted later with minimal churn.

1. **`CoupledSystem` / multi-field assembler (§2.1).** Pure index bookkeeping on
   top of the existing term operators; the natural home would be a
   `pyfvtool.coupling` module. Useful for *any* multi-field / multi-domain model
   (reaction networks, conjugate heat transfer, electro-diffusion), not just the
   GRM.

2. **A first-class `couplingTerm`.** `Assembler.couple` is really a linear
   inter-field source `coeff*(phi_a - phi_b)` — a sibling of the existing
   `linearSourceTerm` / `constantSourceTerm`. As a library primitive it would
   express interphase mass transfer, surface reactions, and Robin-type
   subdomain couplings directly.

3. **A conservative multi-component `storageTerm` (§2.2).** Generalises
   `transientTerm` to a vector of coupled fields with a state-dependent,
   cross-coupled capacity `C_il = d storage_i / d phi_l`, recovering
   `transientTerm` as the scalar diagonal case. This is the most reusable new
   piece of FVM machinery and the strongest upstream candidate.

If/when these move upstream, the repo-local `coupling.py` becomes a thin shim (or
is deleted) and the GRM imports them from PyFVTool instead.
