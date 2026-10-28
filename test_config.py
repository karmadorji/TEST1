from pathlib import Path
from core.config import validate

ROOT = Path(__file__).resolve().parent

def test_config_and_catalog_validate():
    config, labels = validate(ROOT / "config.yaml")
    assert config.recognition.unknown_label == "other/unknown"
    assert any(x.id == "uncertain" and x.kind == "outcome" for x in labels)
    assert abs(sum(config.fusion.weights.values()) - 1.0) < 1e-6
