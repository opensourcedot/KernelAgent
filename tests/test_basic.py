import pytest
from unittest.mock import patch, MagicMock
import os
import sys

def test_imports():
    """Test that main modules can be imported."""
    import triton_kernel_agent
    from triton_kernel_agent import TritonKernelAgent
    from triton_kernel_agent.agent import TritonKernelAgent
    from triton_kernel_agent.manager import WorkerManager
    from triton_kernel_agent.worker import VerificationWorker
    from triton_kernel_agent.prompt_manager import PromptManager
    assert True

def test_prompt_manager_initialization():
    """Test PromptManager initialization."""
    from triton_kernel_agent.prompt_manager import PromptManager
    pm = PromptManager()
    assert pm.templates_dir.exists()
    # Check that templates can be loaded
    assert hasattr(pm, 'env')
    assert pm.env is not None