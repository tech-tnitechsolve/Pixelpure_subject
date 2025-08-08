#!/usr/bin/env python3
"""
🎨 PixelPure Advanced Icon & Logo Generator v2.0
===============================================
Thiết kế icon/logo hiện đại với yếu tố công nghệ cao
- Không có chữ
- Thiết kế tinh gọn, đẹp mắt
- Phong cách tech hiện đại

Tác giả: TNI Tech Solutions
Ngày tạo: August 2025
"""

import os
import math
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

def create_gradient(width, height, color1, color2, direction='vertical'):
    """Tạo gradient background"""
    image = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(image)
    
    if direction == 'vertical':
        for y in range(height):
            ratio = y / height
            r = int(color1[0] + (color2[0] - color1[0]) * ratio)
            g = int(color1[1] + (color2[1] - color1[1]) * ratio)
            b = int(color1[2] + (color2[2] - color1[2]) * ratio)
            a = int(color1[3] + (color2[3] - color1[3]) * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b, a))
    elif direction == 'radial':
        center_x, center_y = width // 2, height // 2
        max_distance = math.sqrt(center_x**2 + center_y**2)
        
        for y in range(height):
            for x in range(width):
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                ratio = min(distance / max_distance, 1.0)
                r = int(color1[0] + (color2[0] - color1[0]) * ratio)
                g = int(color1[1] + (color2[1] - color1[1]) * ratio)
                b = int(color1[2] + (color2[2] - color1[2]) * ratio)
                a = int(color1[3] + (color2[3] - color1[3]) * ratio)
                image.putpixel((x, y), (r, g, b, a))
    
    return image

def create_tech_hexagon(size, center, radius, color, thickness=4):
    """Tạo hình lục giác tech"""
    image = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    points = []
    for i in range(6):
        angle = i * math.pi / 3
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
    
    # Vẽ hexagon outline
    draw.polygon(points, outline=color, width=thickness)
    
    # Vẽ inner lines tạo pattern tech
    for i in range(6):
        angle = i * math.pi / 3
        inner_x = center[0] + (radius * 0.6) * math.cos(angle)
        inner_y = center[1] + (radius * 0.6) * math.sin(angle)
        draw.line([center, (inner_x, inner_y)], fill=color, width=2)
    
    return image

def create_neural_network_pattern(size, center, radius, color):
    """Tạo pattern mạng neural"""
    image = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Tạo các node (điểm)
    nodes = []
    
    # Ring 1 - center
    nodes.append(center)
    
    # Ring 2 - middle
    for i in range(6):
        angle = i * math.pi / 3
        x = center[0] + (radius * 0.4) * math.cos(angle)
        y = center[1] + (radius * 0.4) * math.sin(angle)
        nodes.append((x, y))
    
    # Ring 3 - outer
    for i in range(12):
        angle = i * math.pi / 6
        x = center[0] + (radius * 0.8) * math.cos(angle)
        y = center[1] + (radius * 0.8) * math.sin(angle)
        nodes.append((x, y))
    
    # Vẽ connections
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i < j:
                distance = math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
                if distance < radius * 0.6:
                    # Alpha based on distance
                    alpha = int(255 * (1 - distance / (radius * 0.6)))
                    line_color = (*color[:3], alpha)
                    draw.line([node1, node2], fill=line_color, width=1)
    
    # Vẽ nodes
    for i, node in enumerate(nodes):
        if i == 0:  # Center node
            node_radius = 8
            node_color = (*color[:3], 255)
        elif i <= 6:  # Middle ring
            node_radius = 6
            node_color = (*color[:3], 200)
        else:  # Outer ring
            node_radius = 4
            node_color = (*color[:3], 150)
        
        draw.ellipse([
            node[0] - node_radius, node[1] - node_radius,
            node[0] + node_radius, node[1] + node_radius
        ], fill=node_color)
    
    return image

def create_circuit_pattern(size, center, radius, color):
    """Tạo pattern mạch điện tử"""
    image = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Circuit lines - horizontal và vertical
    line_spacing = radius // 4
    line_width = 3
    
    # Horizontal lines
    for i in range(-2, 3):
        y = center[1] + i * line_spacing
        x1 = center[0] - radius * 0.7
        x2 = center[0] + radius * 0.7
        
        # Main line
        draw.line([(x1, y), (x2, y)], fill=color, width=line_width)
        
        # Circuit components (rectangles)
        comp_width = 20
        comp_height = 8
        comp_x = center[0] + (i * 30)
        draw.rectangle([
            comp_x - comp_width//2, y - comp_height//2,
            comp_x + comp_width//2, y + comp_height//2
        ], outline=color, width=2)
    
    # Vertical connectors
    for i in range(-1, 2):
        x = center[0] + i * line_spacing * 1.5
        y1 = center[1] - radius * 0.7
        y2 = center[1] + radius * 0.7
        
        # Segments với gaps
        segment_length = radius * 0.3
        gap_length = radius * 0.1
        
        current_y = y1
        while current_y < y2:
            end_y = min(current_y + segment_length, y2)
            draw.line([(x, current_y), (x, end_y)], fill=color, width=line_width)
            current_y = end_y + gap_length
    
    # Central processor (square)
    proc_size = radius * 0.4
    draw.rectangle([
        center[0] - proc_size//2, center[1] - proc_size//2,
        center[0] + proc_size//2, center[1] + proc_size//2
    ], outline=color, width=4)
    
    # Corner dots (pins)
    pin_radius = 3
    for corner_x in [center[0] - proc_size//2, center[0] + proc_size//2]:
        for corner_y in [center[1] - proc_size//2, center[1] + proc_size//2]:
            draw.ellipse([
                corner_x - pin_radius, corner_y - pin_radius,
                corner_x + pin_radius, corner_y + pin_radius
            ], fill=color)
    
    return image

def create_modern_tech_icon(size=512):
    """Tạo icon tech hiện đại"""
    # Color palette - Tech gradient
    primary_color = (0, 150, 255, 255)      # Cyan blue
    secondary_color = (138, 43, 226, 255)   # Blue violet  
    accent_color = (255, 20, 147, 255)      # Deep pink
    highlight_color = (255, 255, 255, 255)  # White
    
    center = (size // 2, size // 2)
    
    # Background gradient
    bg = create_gradient(size, size, 
                        (15, 15, 35, 255),    # Dark blue
                        (45, 45, 85, 255),    # Lighter dark blue
                        direction='radial')
    
    # Outer glow
    glow = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_radius = size // 3
    glow_draw.ellipse([
        center[0] - glow_radius, center[1] - glow_radius,
        center[0] + glow_radius, center[1] + glow_radius
    ], fill=(0, 150, 255, 50))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=20))
    
    # Combine background
    bg = Image.alpha_composite(bg, glow)
    
    # Main tech pattern - Neural network style
    main_radius = size // 3
    neural_pattern = create_neural_network_pattern(
        (size, size), center, main_radius, primary_color
    )
    
    # Hexagonal frame
    hex_pattern = create_tech_hexagon(
        (size, size), center, main_radius * 1.2, secondary_color, thickness=6
    )
    
    # Inner circuit pattern (smaller)
    circuit_pattern = create_circuit_pattern(
        (size, size), center, main_radius * 0.6, accent_color
    )
    
    # Central core
    core_radius = main_radius // 4
    core = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    core_draw = ImageDraw.Draw(core)
    
    # Core gradient circle
    for r in range(core_radius, 0, -2):
        alpha = int(255 * (1 - r / core_radius))
        color = (*highlight_color[:3], alpha)
        core_draw.ellipse([
            center[0] - r, center[1] - r,
            center[0] + r, center[1] + r
        ], fill=color)
    
    # Energy rings
    ring_pattern = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    ring_draw = ImageDraw.Draw(ring_pattern)
    
    for i, ring_radius in enumerate([main_radius * 0.9, main_radius * 1.1, main_radius * 1.3]):
        ring_color = (*primary_color[:3], 100 - i * 30)
        ring_draw.ellipse([
            center[0] - ring_radius, center[1] - ring_radius,
            center[0] + ring_radius, center[1] + ring_radius
        ], outline=ring_color, width=2)
    
    # Particles/sparkles effect
    particles = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    particles_draw = ImageDraw.Draw(particles)
    
    import random
    random.seed(42)  # Consistent particles
    
    for _ in range(20):
        # Random position in circle
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(main_radius * 0.8, main_radius * 1.5)
        px = center[0] + distance * math.cos(angle)
        py = center[1] + distance * math.sin(angle)
        
        particle_size = random.randint(2, 6)
        particle_alpha = random.randint(100, 255)
        particle_color = (*highlight_color[:3], particle_alpha)
        
        particles_draw.ellipse([
            px - particle_size, py - particle_size,
            px + particle_size, py + particle_size
        ], fill=particle_color)
    
    # Combine all layers
    result = bg
    result = Image.alpha_composite(result, ring_pattern)
    result = Image.alpha_composite(result, hex_pattern)
    result = Image.alpha_composite(result, neural_pattern)
    result = Image.alpha_composite(result, circuit_pattern)
    result = Image.alpha_composite(result, core)
    result = Image.alpha_composite(result, particles)
    
    return result

def add_text_to_logo(img, text="PixelPure"):
    """Thêm text vào logo (không sử dụng trong version mới)"""
    # Deprecated - không sử dụng text trong icon mới
    return img

def create_icon_sizes(base_img):
    """Tạo các size icon khác nhau"""
    sizes = [16, 32, 48, 64, 128, 256, 512]
    icons = {}
    
    for size in sizes:
        # Resize với anti-aliasing
        icon = base_img.resize((size, size), Image.Resampling.LANCZOS)
        icons[size] = icon
    
    return icons

def save_ico_file(icons, filename="icon.ico"):
    """Lưu file .ico với multiple sizes"""
    # Chuyển đổi sang RGB cho .ico
    rgb_icons = []
    for size, icon in icons.items():
        if icon.mode == 'RGBA':
            # Tạo background đen cho .ico (phù hợp với theme tech)
            rgb_icon = Image.new('RGB', icon.size, (20, 20, 40))
            rgb_icon.paste(icon, mask=icon.split()[-1])  # Use alpha as mask
            rgb_icons.append(rgb_icon)
        else:
            rgb_icons.append(icon)
    
    # Lưu .ico file
    if rgb_icons:
        rgb_icons[0].save(filename, format='ICO', 
                         sizes=[(icon.width, icon.height) for icon in rgb_icons])

def create_logo_variants(base_img):
    """Tạo các variant của logo"""
    variants = {}
    
    # Icon only (main version)
    variants['icon_tech'] = base_img.copy()
    
    # Square version for app stores
    variants['square_app'] = base_img.copy()
    
    # Round version for social media
    round_mask = Image.new('L', base_img.size, 0)
    round_draw = ImageDraw.Draw(round_mask)
    round_draw.ellipse([0, 0, base_img.size[0], base_img.size[1]], fill=255)
    
    round_icon = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
    round_icon.paste(base_img, mask=round_mask)
    variants['round_social'] = round_icon
    
    # Banner version (wide) - minimal tech style
    banner_width, banner_height = 1024, 256
    banner = create_gradient(banner_width, banner_height,
                           (15, 15, 35, 255),
                           (45, 45, 85, 255),
                           direction='horizontal')
    
    # Add multiple small icons across banner
    icon_size = 180
    logo_resized = base_img.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
    
    # Center position
    x_pos = (banner_width - icon_size) // 2
    y_pos = (banner_height - icon_size) // 2
    banner.paste(logo_resized, (x_pos, y_pos), logo_resized)
    
    variants['banner_tech'] = banner
    
    return variants

def main():
    """Main function"""
    print("🎨 PixelPure Advanced Logo & Icon Generator v2.0")
    print("=" * 50)
    
    # Tạo thư mục assets
    os.makedirs("assets", exist_ok=True)
    os.makedirs("assets/icons", exist_ok=True)
    os.makedirs("assets/logos", exist_ok=True)
    
    print("🔧 Tạo icon tech hiện đại...")
    base_logo = create_modern_tech_icon(512)
    
    print("🎯 Tạo các variant...")
    variants = create_logo_variants(base_logo)
    
    print("📐 Tạo icon sizes...")
    icon_sizes = create_icon_sizes(base_logo)
    
    print("💾 Lưu files...")
    
    # Lưu logo variants
    for name, img in variants.items():
        img.save(f"assets/logos/{name}.png", "PNG")
        print(f"  ✅ assets/logos/{name}.png")
    
    # Lưu icon sizes
    for size, icon in icon_sizes.items():
        icon.save(f"assets/icons/icon_{size}x{size}.png", "PNG")
        print(f"  ✅ assets/icons/icon_{size}x{size}.png")
    
    # Lưu .ico file cho Windows
    save_ico_file(icon_sizes, "assets/icon.ico")
    print(f"  ✅ assets/icon.ico")
    
    # Copy icon.ico to root cho PyInstaller
    # Convert to RGB with tech background
    ico_base = icon_sizes[256].copy()
    rgb_ico = Image.new('RGB', ico_base.size, (20, 20, 40))  # Dark tech background
    rgb_ico.paste(ico_base, mask=ico_base.split()[-1])
    rgb_ico.save("icon.ico", "ICO")
    print(f"  ✅ icon.ico (for PyInstaller)")
    
    print("\n🎉 Logo và icon tech đã được tạo thành công!")
    print("\n📁 Files đã tạo:")
    print("  📄 icon.ico                    - Icon cho PyInstaller (tech theme)")
    print("  📁 assets/logos/               - Logo variants (tech style)")
    print("  📁 assets/icons/               - Icon sizes (modern tech)")
    print("  📄 assets/icon.ico             - Windows icon file")
    print("\n🎨 Design Features:")
    print("  ✨ Neural network pattern")
    print("  🔷 Hexagonal tech frame") 
    print("  ⚡ Circuit board elements")
    print("  🌟 Energy particles effect")
    print("  🎯 No text - pure icon design")
    print("  🚀 High-tech aesthetic")

if __name__ == "__main__":
    main()
