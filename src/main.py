"""
Main entry point for the modernized PyQt application.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.presentation.gui.modern_main_window import create_application, ModernMainWindow as MainWindow


def main():
    """Main application entry point."""
    try:
        # Create application
        app = create_application()
        
        # Create and show main window
        window = MainWindow()
        window.show()
        
        # Start event loop
        return app.exec_()
        
    except Exception as e:
        import traceback
        print(f"Failed to start application: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())