"""Check the research environment without downloading data or model weights."""

import importlib
import importlib.metadata
import json
from pathlib import Path
import site
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    expected_prefix = root / ".venv"
    assert Path(sys.prefix).resolve() == expected_prefix.resolve(), (
        "Run with the repository .venv/Scripts/python.exe"
    )
    assert sys.prefix != sys.base_prefix, "A virtual environment is required"
    assert not site.ENABLE_USER_SITE, "User site-packages must be disabled"
    assert "include-system-site-packages = false" in (
        expected_prefix / "pyvenv.cfg"
    ).read_text(encoding="utf-8")

    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "torch": "torch",
        "einops": "einops",
        "huggingface_hub": "huggingface-hub",
        "matplotlib": "matplotlib",
        "tqdm": "tqdm",
        "safetensors": "safetensors",
        "ccxt": "ccxt",
        "sklearn": "scikit-learn",
        "yaml": "PyYAML",
        "pytest": "pytest",
    }
    versions = {}
    for module_name, distribution_name in packages.items():
        module = importlib.import_module(module_name)
        assert Path(module.__file__).resolve().is_relative_to(expected_prefix), (
            f"{module_name} was loaded outside .venv: {module.__file__}"
        )
        versions[distribution_name] = importlib.metadata.version(distribution_name)

    sys.path.insert(0, str(root))
    from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: F401
    import numpy as np
    from sklearn.linear_model import Ridge
    import torch
    import torch.nn.functional as functional

    x = np.arange(20, dtype=np.float64).reshape(10, 2)
    assert np.isfinite(Ridge().fit(x, x[:, 0]).predict(x)).all()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(1, 2, 8, 16, device=device)
    attention = functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    assert torch.isfinite(attention).all().item()
    if device == "cuda":
        torch.cuda.synchronize()

    print(json.dumps({
        "status": "passed",
        "executable": sys.executable,
        "base_interpreter": sys.base_prefix,
        "python": sys.version.split()[0],
        "isolated_packages": True,
        "packages": versions,
        "torch_cuda_build": torch.version.cuda,
        "smoke_device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }, indent=2))


if __name__ == "__main__":
    main()
