"""Artifact-only reporting: never imports or executes experiment runners."""

from ifcred.reporting.report import generate_report

__all__ = ["generate_report"]
