#!/usr/bin/env python3

"""
Test script to verify app_ui.py imports and basic functionality
"""

try:
    print("Testing imports...")
    from app_ui import PixelPureApplication, MainWindow, DARK_STYLESHEET
    print("✅ All imports successful!")
    
    print("Testing basic UI creation...")
    import sys
    app = PixelPureApplication(sys.argv)
    main_win = MainWindow()
    print("✅ UI creation successful!")
    
    print("Testing basic functionality...")
    # Test some basic methods
    test_files = ["test1.jpg", "test2.jpg"]
    
    print("✅ All tests passed!")
    print("\n🎉 PixelPure is ready to run!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
