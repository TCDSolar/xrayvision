Xrayvision 0.2.1 (2025-04-04)
=============================

Backwards Incompatible Changes
------------------------------

- Optional parameters are now keyword only for the :mod:`xrayvision.transform`, :mod:`xrayvision.imaging` and :mod:`xrayvision.visibility` modules.
  Remove ``natural`` keyword in favour of ``scheme`` keyword which can be either 'natural' or 'uniform'. (`#58 <https://github.com/TCDSolar/xrayvision/pull/58>`__)
- Make the software compatible with :class:`~xrayvision.visibility.Visibilities`. (`#67 <https://github.com/TCDSolar/xrayvision/pull/67>`__)


Features
--------

- Introduce new API for handling visibilities: `~xrayvision.visibility.VisibilitiesABC` and `~xrayvision.visibility.VisMetaABC`, along with implemented versions of the classes `~xrayvision.visibility.Visibilities` and `~xrayvision.visibility.VisMeta`. (`#55 <https://github.com/TCDSolar/xrayvision/pull/55>`__)
- Add equality operator to `~xrayvision.visibility.Visibilities` and `~xrayvision.visibility.VisMeta`. (`#64 <https://github.com/TCDSolar/xrayvision/pull/64>`__)
- Add ``__getitem__`` and `~xrayvision.visibility.Visibilities.index_by_label` methods to slice visibilities by index or based on their label. (`#65 <https://github.com/TCDSolar/xrayvision/pull/65>`__)
- Add `~xrayvision.coordinates.frames.Projective` coordinate frame to represent generic observer based projective coordinate system. (`#76 <https://github.com/TCDSolar/xrayvision/pull/76>`__)
- Enable users to manually set total flux/counts required by `~xrayvision.mem.mem`. (`#78 <https://github.com/TCDSolar/xrayvision/pull/78>`__)


Bug Fixes
---------

- Fix a bug where the x, y dimensions were not being treated consistently in :mod:`xrayvision.transform`. (`#58 <https://github.com/TCDSolar/xrayvision/pull/58>`__)
- Change typing of meta input to `~xrayvision.visibility.Visibilities` to be a `~xrayvision.visibilities.VisMeta`. (`#63 <https://github.com/TCDSolar/xrayvision/pull/63>`__)
- Fix bug when creating :class:`~xrayvision.visibility.Visibilities` with default meta. (`#66 <https://github.com/TCDSolar/xrayvision/pull/66>`__)
- Fix bug introduced in recent refactor where images were transposed due to array vs cartesian array indexing. (`#73 <https://github.com/TCDSolar/xrayvision/pull/73>`__)
- Fix a bug reintroduced in `~xrayvision.mem.mem` which caused the output to be transposed incorrectly. (`#74 <https://github.com/TCDSolar/xrayvision/pull/74>`__)


Improved Documentation
----------------------

- Update README fix badges and remove old example. (`#53 <https://github.com/TCDSolar/xrayvision/pull/53>`__)
- Update RHESSI example to use the same image dimensions and pixel size throughout. (`#74 <https://github.com/TCDSolar/xrayvision/pull/74>`__)


Trivial/Internal Changes
------------------------

- Fix small bug in isort configuration. (`#53 <https://github.com/TCDSolar/xrayvision/pull/53>`__)
- Format code with ruff and turn on ruff format in pre-commit. (`#61 <https://github.com/TCDSolar/xrayvision/pull/61>`__)
- Update README with useful links to example and issue tracker. (`#70 <https://github.com/TCDSolar/xrayvision/pull/70>`__)
- Update project configuration move to `project.toml` and individual config files for each tool (isort, ruff, pytest, etc) and add zenodo config file and mailmap. (`#81 <https://github.com/TCDSolar/xrayvision/pull/81>`__)


Xrayvision 0.1.dev60+g6fe95fa.d20210925 (2021-09-25)
====================================================

No significant changes.
