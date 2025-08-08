#!/usr/bin/env python3
"""
🎨 PixelPure Minimal Icon Generator
==================================
Tạo icon tối giản cho system tray và taskbar

Tác giả: TNI Tech Solutions
Ngày tạo: August 2025
"""

import os
import math
from PIL import Image, ImageDraw

def create_minimal_tech_icon(size=64):
    """Tạo icon tối giản cho system tray"""
    # Dark theme colors
    bg_color = (25, 25, 45, 255)       # Dark blue
    primary_color = (0, 150, 255, 255)  # Cyan blue
    accent_color = (255, 255, 255, 255) # White
    
    # Create base image
    img = Image.new('RGBA', (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    
    center = (size // 2, size // 2)
    
    # Main circle (simplified neural core)
    main_radius = size // 3
    draw.ellipse([
        center[0] - main_radius, center[1] - main_radius,
        center[0] + main_radius, center[1] + main_radius
    ], outline=primary_color, width=3)
    
    # Inner connections (simplified)
    inner_radius = main_radius // 2
    for i in range(4):
        angle = i * math.pi / 2
        x = center[0] + inner_radius * math.cos(angle)
        y = center[1] + inner_radius * math.sin(angle)
        
        # Connection line
        draw.line([center, (x, y)], fill=primary_color, width=2)
        
        # End point
        draw.ellipse([x-2, y-2, x+2, y+2], fill=accent_color)
    
    # Center core
    core_radius = 4
    draw.ellipse([
        center[0] - core_radius, center[1] - core_radius,
        center[0] + core_radius, center[1] + core_radius
    ], fill=accent_color)
    
    return img

def create_monochrome_icon(size=32):
    """Tạo icon đơn sắc cho small sizes"""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    center = (size // 2, size // 2)
    color = (255, 255, 255, 255)
    
    # Simple geometric design
    radius = size // 3
    
    # Outer ring
    draw.ellipse([
        center[0] - radius, center[1] - radius,
        center[0] + radius, center[1] + radius
    ], outline=color, width=2)
    
    # Cross lines
    line_length = radius - 2
    draw.line([
        center[0] - line_length, center[1],
        center[0] + line_length, center[1]
    ], fill=color, width=2)
    
    draw.line([
        center[0], center[1] - line_length,
        center[0], center[1] + line_length
    ], fill=color, width=2)
    
    # Center dot
    draw.ellipse([
        center[0] - 2, center[1] - 2,
        center[0] + 2, center[1] + 2
    ], fill=color)
    
    return img

def main():
    """Main function"""
    print("🎨 PixelPure Minimal Icon Generator")
    print("=" * 40)
    
    # Create minimal icons for system use
    os.makedirs("assets/minimal", exist_ok=True)
    
    # System tray icon (16x16, 32x32)
    for size in [16, 24, 32, 48]:
        if size <= 32:
            icon = create_monochrome_icon(size)
        else:
            icon = create_minimal_tech_icon(size)
        
        icon.save(f"assets/minimal/minimal_{size}x{size}.png", "PNG")
        print(f"  ✅ assets/minimal/minimal_{size}x{size}.png")
    
    # Create ICO file with minimal design
    sizes = [16, 24, 32, 48]
    icons = []
    
    for size in sizes:
        if size <= 32:
            icon = create_monochrome_icon(size)
        else:
            icon = create_minimal_tech_icon(size)
        
        # Convert to RGB with dark background
        rgb_icon = Image.new('RGB', (size, size), (25, 25, 45))
        rgb_icon.paste(icon, mask=icon.split()[-1] if icon.mode == 'RGBA' else None)
        icons.append(rgb_icon)
    
    # Save minimal ICO
    if icons:
        icons[0].save("assets/minimal_icon.ico", format='ICO',
                     sizes=[(icon.width, icon.height) for icon in icons])
        print(f"  ✅ assets/minimal_icon.ico")
    
    print("\n🎉 Minimal icons created!")
    print("💡 Use minimal_icon.ico for system tray applications")

if __name__ == "__main__":
    main()
