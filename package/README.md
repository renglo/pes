# PES Handlers Module

Custom handlers for the PES application, packaged as a proper Python library.

## Overview

This module provides handlers and utilities for PES-specific functionality.

## Installation

### For Local Development

```bash
# Clone or navigate to the repository
cd /path/to/extensions/pes

# Install in editable mode
pip install -e package/
```

## Usage

### Basic Handler Usage

All handlers implement a standard `run(payload)` interface:

```python
from pes.handlers import ExampleHandler

# Example Handler
handler = ExampleHandler()
result = handler.run({'key': 'value'})
print(result)
```

### Dynamic Handler Loading

Use the built-in registry to load handlers dynamically by name:

```python
from pes import get_handler, list_handlers

# List all available handlers
print(list_handlers())

# Load a handler by name
HandlerClass = get_handler('example_handler')
handler = HandlerClass()
result = handler.run({'key': 'value'})
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e "package/[dev]"

# Run tests
pytest package/tests/
```

### Adding New Handlers

1. Create a new handler file in `pes/handlers/`:

```python
# pes/handlers/my_handler.py
class MyHandler:
    def __init__(self, config=None):
        self.config = config or {}
    
    def run(self, payload):
        # Implementation
        return {'success': True, 'output': result}
```

2. Register it in `pes/__init__.py`:

```python
def get_my_handler():
    from pes.handlers.my_handler import MyHandler
    return MyHandler

HANDLERS = {
    # ... existing handlers
    'my_handler': get_my_handler,
}
```

3. Add to `pes/handlers/__init__.py`:

```python
from pes.handlers.my_handler import MyHandler

__all__ = [
    # ... existing handlers
    'MyHandler',
]
```

## Architecture

This module follows the refactored architecture:

- **Framework Agnostic**: Handlers don't depend on Flask or any web framework
- **Config Injection**: Configuration is passed via constructor, not imported globally
- **Standardized Interface**: All handlers implement `run(payload) → result`
- **Proper Packaging**: Uses modern Python packaging (`pyproject.toml`)
- **Dynamic Loading**: Registry-based handler discovery and instantiation

## License

See main repository LICENSE files.




