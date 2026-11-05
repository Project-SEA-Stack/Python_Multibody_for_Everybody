Validation
==========

Validation has been migrated from standalone scripts to a self-contained
pytest suite under ``tests/``.  See the :ref:`Tests <Developer's documentation>`
section for the full description of the four test suites (MATLAB symbolic,
MATLAB energy, Chrono regression, YAML parser, and linearization) and
instructions for running them.

Flexible-model validation
-------------------------

The flexible suite under ``tests/cases/flexible/`` separates test inputs,
stored results, and manual reference-generation scripts:

* ``models/`` holds frozen copies of the tapered pendulum and moderate
  double pendulum models. ``regression_catalog.py`` points to these fixtures,
  not the editable models under ``Examples_flexible_FSM/``.
* ``regression_runner.py`` loads each fixture, builds an ``MbdSystem``,
  integrates it, and compares positions, velocities, and angles against the
  ``.npy`` baselines in ``truth/<case_id>/``.
* The regression test and standalone runner share an NRMSE threshold of
  ``1e-4``. The models use Radau with ``rtol=1e-4`` and ``atol=1e-6``;
  solver tolerances and trajectory NRMSE are distinct measures.
* ``pychrono_backup/`` holds the five manual PyChrono reference scripts.
  Pytest does not import or execute these scripts. The frequency
  cross-validation test uses previously recorded, hard-coded frequencies.
  Three scripts write trajectory ``.npz`` files into ``truth/``; these files
  are retained as reference results but are not currently loaded by pytest.

Run the flexible suite from the repository root::

   python -m pytest tests/cases/flexible -q

Editing a user-facing example does not change its frozen regression model.
Only update a fixture and its baseline deliberately, after independent
validation. Regenerate the FSM baseline with::

   python -m tests.cases.flexible.regression_runner --update <case_id>
