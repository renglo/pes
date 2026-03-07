#!/usr/bin/env python3
"""
Example usage of PES handlers
Demonstrates how to use handlers with sample data
"""

import json
from pes_noma import get_handler, list_handlers


def test_dynamic_loading():
    """Test dynamic handler loading"""
    print("=" * 60)
    print("Testing Dynamic Handler Loading")
    print("=" * 60)

    # List all handlers
    print("\nAvailable handlers:")
    handlers = list_handlers()
    if handlers:
        for handler_name in handlers:
            print(f"  - {handler_name}")
    else:
        print("  No handlers registered yet.")
        print("  Add handlers to pes_noma/handlers/ and register them in pes_noma/__init__.py")


if __name__ == '__main__':
    print("\nPES Noma Handlers - Example Usage\n")

    # Run tests
    test_dynamic_loading()

    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)
