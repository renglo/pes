"""
PES Handlers

All custom handlers for the PES application.
Each handler should implement a `run(payload)` method.

Note: Handlers are imported lazily to avoid import errors when
optional dependencies are not installed.
"""

# Add your handler exports here
__all__ = [
    # 'ExampleHandler',
]

def __getattr__(name):
    """Lazy import handlers when accessed."""
    # Add your handler imports here
    # if name == 'ExampleHandler':
    #     from pes.handlers.example_handler import ExampleHandler
    #     return ExampleHandler
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")




