# Technical report audit notes

Delivery: user-selected Markdown, not HTML/MCP publishing. Audience: technical.
The requested Markdown format overrides the report skill's generic delivery default.

## Report structure mapping

Technical summary: section 1. Scope/definitions: sections 1–3 and 7, before
interpreting the corresponding evidence. Model and optimization: sections 2–4.
Real-ray evidence: section 5. Joint coherent model and visual evidence: section 6.
Robustness: section 7. Native POP is separated in section 8. Reproducibility:
section 9. Limitations/next steps/open questions: section 10, plus local caveats.
The engineering workflow requires definitions and optical construction before
the detailed visual evidence, so the generic technical report order is adjusted.

## Canonical sources

- Supplied 10X_OBJ_0.5na-2.zmx, preserved copy and SHA-256 in source_audit.json.
- Local OpticStudio 2022 R2.02, standalone ZOS-API, not the user's GUI instance.
- ZOSAPI_Interfaces.dll/XML from the installed OpticStudio version; Standard=65.
- Native ray coordinates/OPD: rays_field_*.npz, sine_fan.csv,
  diffraction_order_calibration.csv, layout_rays.csv.
- Actual UDA ray-blocking checks: uda_ray_blocking.csv, uda_validation.json.
- Analytic-reference formulas and API semantics: linked Ansys primary sources
  near the relevant paragraphs of the report; no invented references.
- Coherent filtering: src/real_objective_coherent.py, exported JSON and NPZ.
- Native POP: native_pop/*.npz, .txt, .CFG, .json. Never conflate these with
  Python coherent reconstruction. Early failed probes remain in diagnostics.

## Chart contracts

1. real_lens_layout.png: geometric relationship/actual ray path plot. Coordinates
   are traced by Zemax, not generated as an illustration. Two panels distinguish
   complete assembly and enlarged objective. Physical mm; note unequal scales.
2. real_sine_condition.png: ordered-aperture error curves, 401 meridional samples
   per field, separate Fourier-coordinate and finite-conjugate errors. Blue,
   orange, olive with distinct line patterns; temporary +/-0.1% reference.
3. real_wavefront.png: matrix/pupil error maps, 3209 rays at each of 3 fields,
   same -0.4..0.4 wave scale, piston only removed. Diverging blue/white/gold.
4. real_fourier_spectrum.png: physical-frequency heatmaps before/after mask,
   common pre-mask peak normalization and -8..0 log scale. Nominal coordinate
   labels; mask itself evaluated using actual measured coordinate mapping.
5. real_filter_comparison.png: paired amplitude/intensity images in the central
   nominal 10x image crop. Same scale within open/filtered columns; explicitly
   different scales across columns. Adjacent report text explains this.
6. native_pop_comparison.png: native irradiance in W/mm2. Physical image axes;
   common absolute scale within each open/filtered column. These arrays are
   separate from the Python joint-model evidence.
7. native_fourier_plane.png: native S25 irradiance with the same-plane ideal
   mask product. The second panel is explicitly labeled derived, not another
   native POP execution. Native filtered propagation is separately exported at S28.

Exact surface tables, optimizer terms, diffraction-order locations and a few
sensitivity cases use tables because exact engineering lookup is the purpose.

## QA scope

Scientific figures are rendered in the Anaconda fourier-filter environment and
visually inspected. Math uses $...$ and standalone $$...$$ in Markdown; formulas
inside table cells avoid literal vertical bars. Original and copied source hashes
are compared. Original geometry reconstruction, native ray success, sine residuals,
UDA masking, lens edge thickness and numerical sensitivity are regression-tested.
No claim is made of physical hardware qualification, vector diffraction validity,
full-field shift invariance, or broadband UV suitability.

Native 2048 vs 4096 relative suppression changes 0.454 dB, but total unfiltered
power changes 10.000%. This is NOT classified as full absolute-field convergence.
The report states this limitation and suggests controlling source pixel-area
sampling before precision throughput or final rejection-ratio claims.

Rejected early two-objective/negative-gap attempt is isolated in
diagnostics/rejected_pair; it is not a deliverable optical system.
