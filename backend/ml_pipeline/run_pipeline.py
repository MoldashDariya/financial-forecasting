"""Compatibility shim.

This module now delegates to the methodology-aligned universal pipeline
implemented in `backend/services/ml_pipeline.py`.
"""

try:
    from services.ml_pipeline import load_company_csv, pipeline_dependency_status, run_pipeline
except ModuleNotFoundError:
    from backend.services.ml_pipeline import load_company_csv, pipeline_dependency_status, run_pipeline
