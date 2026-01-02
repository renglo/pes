"""
PES Handlers Package

This package provides custom handlers for the PES application.

Usage:
    from pes.handlers import ExampleHandler
    
    # Use a handler
    handler = ExampleHandler()
    result = handler.run(payload)
"""

__version__ = "1.0.0"
__all__ = []

# Registry of available handlers
# Add your handlers here as you create them
HANDLERS = {
    # 'example_handler': get_example_handler,
}

def get_handler(handler_name: str):
    """
    Get a handler class by name.
    
    Args:
        handler_name: Name of the handler (e.g., 'example_handler')
        
    Returns:
        Handler class (not instantiated)
        
    Raises:
        KeyError: If handler_name is not found
        
    Example:
        handler_class = get_handler('example_handler')
        handler = handler_class()
        result = handler.run(payload)
    """
    if handler_name not in HANDLERS:
        available = ', '.join(HANDLERS.keys())
        raise KeyError(f"Handler '{handler_name}' not found. Available handlers: {available}")
    
    return HANDLERS[handler_name]()

def list_handlers():
    """
    List all available handlers.
    
    Returns:
        List of handler names
    """
    return list(HANDLERS.keys())




