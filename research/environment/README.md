# Kronos research Python environment

The Windows research environment lives in the repository `.venv` directory.
It uses Python 3.11.14 and PyTorch 2.7.1 with CUDA 11.8. Third-party packages
are installed separately from the existing Conda environment and user packages.
The original repository requirements are retained; `requirements.in` adds the
OKX client, linear-model tooling, YAML support and pytest for the first research phase.

This venv uses the base interpreter at `E:\conda\envs\kronos`. Keep that
interpreter directory in place. The package isolation does not make the Python
runtime self-contained or the environment movable.

## Use

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python research\environment\verify_environment.py
python -m pip check
```

Activation is optional. Commands can always select the interpreter explicitly:

```powershell
& '.\.venv\Scripts\python.exe' examples\prediction_okx_btcusdt.py --help
& '.\.venv\Scripts\python.exe' research\environment\verify_environment.py
```

Use this interpreter for subsequent research commands. The environment is
ignored by Git; the input requirements, complete version lock and verification
record are kept under `research/environment`.

## Recreate

For another Windows machine, replace the base Python path with an installed
Python 3.11 interpreter. Create a fresh environment at the destination rather
than copying `.venv`. These commands run from the repository root:

```powershell
& 'E:\conda\envs\kronos\python.exe' -m venv --copies --prompt kronos-research .venv
& '.\.venv\Scripts\python.exe' -m pip --isolated install --index-url https://pypi.org/simple pip==25.2
& '.\.venv\Scripts\python.exe' -m pip --isolated install --index-url https://download.pytorch.org/whl/cu118 'torch==2.7.1+cu118'
& '.\.venv\Scripts\python.exe' -m pip --isolated install --index-url https://pypi.org/simple -r research\environment\requirements.lock.txt
& '.\.venv\Scripts\python.exe' -m pip check
& '.\.venv\Scripts\python.exe' research\environment\verify_environment.py
```

Install PyTorch from its official CUDA index first; the PyPI-only lock command
then retains that exact installed build. `requirements.lock.txt` pins installed
distribution versions, including transitive dependencies and pip/setuptools;
it is a Windows/Python 3.11 version lock, not a cross-platform or hash lock.

To intentionally resolve new dependency versions, install `requirements.in`
after PyTorch, validate, and regenerate the lock with this environment's
`python -m pip freeze --all`. Review the resulting changes before using them
for a registered experiment.

## Validation scope

`verify_environment.py` checks package isolation, required imports, a Ridge
fit and a PyTorch attention operation on the available device. It does not
download models, call an exchange or evaluate strategy returns.
`verification.json` records the successful setup checks and the separate
offline model/regression checks performed during environment creation.

References: [Python venv documentation](https://docs.python.org/3.11/library/venv.html)
and [PyTorch 2.7.1 installation commands](https://pytorch.org/get-started/previous-versions/#v271).
