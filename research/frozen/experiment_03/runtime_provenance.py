"""Record Windows package RECORDs and numerical runtime binaries as metadata.

No market source, model weights, credentials, fitting, or holdout access occurs.
The separately pinned LightGBM dependency is covered by dependencies.json.
"""
from __future__ import annotations
import importlib
import importlib.metadata
from pathlib import Path
import sys
from research.frozen.experiment_03 import guard

PACKAGES = ('numpy', 'scipy', 'scikit-learn', 'pandas', 'torch',
            'requests', 'PyYAML', 'threadpoolctl')
NATIVE_SUFFIXES = ('.pyd', '.dll')


def main():
    root = guard.ROOT.resolve()
    environment = (root / '.venv').resolve()
    executable = Path(sys.executable).resolve()
    if not executable.is_relative_to(environment):
        raise ValueError('Use the explicit repository .venv interpreter')
    output = guard.RUN / 'provenance/runtime_native_files.json'
    if output.exists():
        raise FileExistsError('Runtime provenance is immutable; do not overwrite')
    for name in ('numpy', 'scipy.optimize', 'scipy.linalg', 'sklearn.preprocessing', 'torch'):
        importlib.import_module(name)
    candidates = {}

    def include(path, purpose):
        path = Path(path).resolve()
        if not path.is_relative_to(environment) or not path.is_file():
            raise ValueError('Runtime file must be inside repository .venv: ' + str(path))
        relative = path.relative_to(root).as_posix()
        candidates.setdefault(relative, {'path': path, 'purposes': set()})['purposes'].add(purpose)

    include(executable, 'explicit research interpreter')
    versions = {}
    for name in PACKAGES:
        distribution = importlib.metadata.distribution(name)
        versions[name] = distribution.version
        records = [item for item in distribution.files or []
                   if item.name == 'RECORD' and item.parent.name.endswith('.dist-info')]
        if len(records) != 1:
            raise ValueError('Exactly one installed package RECORD required: ' + name)
        include(distribution.locate_file(records[0]), name + ' installed RECORD')
    excluded_modules = []
    for module_name, module in sorted(list(sys.modules.items())):
        file = getattr(module, '__file__', None)
        if file and Path(file).suffix.lower() in NATIVE_SUFFIXES:
            if Path(file).resolve().is_relative_to(environment):
                include(file, 'loaded native module ' + module_name)
            elif module_name.split('.')[0] in ('numpy', 'scipy', 'sklearn', 'pandas', 'torch'):
                raise ValueError('Numerical native module loaded outside repository .venv: ' + module_name)
            else:
                # A venv can use its base interpreter's stdlib extension modules.
                # Their external files are not read under this bounded task.
                excluded_modules.append(module_name)
    site_packages = environment / 'Lib/site-packages'
    for directory_name in ('numpy.libs', 'scipy.libs', 'torch/lib'):
        directory = site_packages / directory_name
        if not directory.is_dir():
            raise ValueError('Expected numerical native runtime directory missing: ' + directory_name)
        for file in sorted(directory.glob('*.dll')):
            include(file, directory_name + ' native runtime DLL')
    files, total = {}, 0
    for index, (relative, entry) in enumerate(sorted(candidates.items()), start=1):
        path = entry['path']
        size = path.stat().st_size
        files[relative] = {'sha256': guard.sha(path), 'bytes': size,
                           'purposes': sorted(entry['purposes'])}
        total += size
        if index % 10 == 0 or index == len(candidates):
            print('Hashed metadata/native files ' + str(index) + '/' + str(len(candidates)), flush=True)
    result = {'schema_version': 1, 'created_at_utc': guard.now(),
              'sys_executable': executable.relative_to(root).as_posix(),
              'python_version': sys.version, 'versions': versions,
              'files_count': len(files), 'total_bytes': total, 'files': files,
              'scope': 'Package installation RECORDs, loaded Python native modules, numpy/scipy DLLs and all torch/lib DLLs; metadata-only hashes under ROOT/.venv',
              'lightgbm': 'Separately pinned Python/native files covered by provenance/dependencies.json; not installed or modified here',
              'excluded_loaded_native_module_names_outside_venv': excluded_modules,
              'exclusion_limit': 'Base-interpreter stdlib extensions outside ROOT/.venv are listed by module name only; no external file bytes read or hashed',
              'market_or_holdout_values_accessed': False, 'fits_performed': 0}
    guard.new_json(output, result)
    print('Runtime files=' + str(len(files)) + ' bytes=' + str(total)
          + ' JSON_SHA256=' + guard.sha(output), flush=True)
    return result


if __name__ == '__main__':
    main()
