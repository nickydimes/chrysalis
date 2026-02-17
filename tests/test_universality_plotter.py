import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests
import matplotlib.pyplot as plt

from src.tools.universality_plotter import _load_image


class TestLoadImage:
    def test_loads_existing_image(self, tmp_path):
        """Create a small PNG and verify _load_image renders it."""
        img_path = tmp_path / "test.png"
        fig_tmp, ax_tmp = plt.subplots()
        ax_tmp.plot([0, 1], [0, 1])
        fig_tmp.savefig(img_path)
        plt.close(fig_tmp)

        fig, ax = plt.subplots()
        _load_image(ax, str(img_path), "Test Title")
        assert ax.get_title() == "Test Title"
        assert len(ax.images) == 1
        plt.close(fig)

    def test_missing_image_shows_placeholder(self, tmp_path):
        """Missing file should show placeholder text, not raise."""
        fig, ax = plt.subplots()
        _load_image(ax, str(tmp_path / "nonexistent.png"), "Missing")
        assert ax.get_title() == "Missing"
        assert len(ax.images) == 0
        assert len(ax.texts) > 0
        plt.close(fig)
