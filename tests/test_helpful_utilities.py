#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `helpful_utilities` package."""


import unittest
from click.testing import CliRunner

from helpful_utilities import helpful_utilities
from helpful_utilities import cli


class TestHelpful_utilities(unittest.TestCase):
    """Tests for `helpful_utilities` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'helpful_utilities.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
