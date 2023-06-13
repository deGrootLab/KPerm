# Changelog

## [0.X.X] - 2023-0X-XX

### Added
- set_coord() and set_trajs() for specifying coordinate and trajectory files.
- Users now can adjust the radius of the binding site using keyword `bs_radius` in `Channel.run()`. Default: 4.0 (Angstrom)

### Changed
- findCycles renamed to find_cycle
- permeationMFPT renamed to mfpt
- plotCycles renamed to plot_cycle
- plotNetFlux renamed to plotNetFlux
- computeStats renamed to compute_stats

### Changed

## [0.1.1] - 2023-05-23

### Changed

- Removed 1 Angstrom buffer for S0 to improve accuracy of S0 occupancy determination.

## [0.1.0] - 2023-05-19

### Added

- Calculation of net fluxes for permeation cycles
- Method of calculating current via counting permeation events across S2-S3 interface.
- Option of writing indices of objects in SF or involved in permeation to files.
- Changelog.

### Changed

- Charge-scaled tutorial.
- Part of the codes formatted to conform to the PEP 8 style.

### Removed

- Obsolete methods for determining SF occupancy and ion jumps in SF.

## [0.0.1] - 2023-03-30

### Added

- Initial release of KPERM

[0.1.1]: https://github.com/deGrootLab/KPerm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/deGrootLab/KPerm/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/deGrootLab/KPerm/releases/tag/v0.0.1
