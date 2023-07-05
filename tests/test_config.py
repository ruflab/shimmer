from utils import PROJECT_DIR

from shimmer.config import load_config


def test_config():
    config = load_config(PROJECT_DIR / "config")
    assert "__shimmer__" in config
