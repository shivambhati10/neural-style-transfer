import random
import os

def generate_ascii_art(width=30, height=15, style="default"):
    """
    Generate a simple ASCII art image based on a style.
    
    Args:
        width (int): Width of the image
        height (int): Height of the image
        style (str): Style to use ("default", "geometric", "sketch", "dots")
        
    Returns:
        list: ASCII art as a list of strings
    """
    # Different character sets for different styles
    styles = {
        "default": ".,-~:;=!*#$@",
        "geometric": "+|/-\\[]{}",
        "sketch": "._-=+*^%",
        "dots": ".,:;",
        "blocks": "░▒▓█"
    }
    
    # Use requested style or default to mixed characters
    chars = styles.get(style, ".,-~:;=!*#$@")
    
    # Generate the image
    image = []
    for _ in range(height):
        row = ''.join(random.choice(chars) for _ in range(width))
        image.append(row)
    
    return image

def display_ascii_art(image):
    """Display ASCII art with a border."""
    if not image:
        print("Empty image")
        return
    
    width = len(image[0])
    print("+" + "-" * width + "+")
    for row in image:
        print("|" + row + "|")
    print("+" + "-" * width + "+")

def create_house_ascii():
    """Create a simple house in ASCII art."""
    return [
        "    /\\    ",
        "   /  \\   ",
        "  /____\\  ",
        "  |    |  ",
        "  |    |  ",
        "  |____|  "
    ]

def mix_styles(content_image, style="default", style_strength=0.5):
    """
    Mix content image with a style.
    
    Args:
        content_image (list): Content image as a list of strings
        style (str): Style to apply
        style_strength (float): How strongly to apply the style (0.0-1.0)
        
    Returns:
        list: Styled image as a list of strings
    """
    # Different character sets for different styles
    styles = {
        "default": ".,-~:;=!*#$@",
        "geometric": "+|/-\\[]{}",
        "sketch": "._-=+*^%",
        "dots": ".,:;",
        "blocks": "░▒▓█"
    }
    
    # Use requested style or default to mixed characters
    style_chars = styles.get(style, ".,-~:;=!*#$@")
    
    # Determine size
    height = len(content_image)
    width = max(len(row) for row in content_image)
    
    # Normalize content image width
    content_norm = [row.ljust(width) for row in content_image]
    
    # Create result image
    result = []
    for row in content_norm:
        new_row = ""
        for char in row:
            # Empty space stays empty
            if char == " ":
                new_row += " "
            else:
                # Apply style based on style_strength
                if random.random() < style_strength:
                    new_row += random.choice(style_chars)
                else:
                    new_row += char
        result.append(new_row)
    
    return result

def simple_animation(content_image, styles=None, frames=5):
    """
    Create a simple animation showing style transfer progress.
    
    Args:
        content_image (list): Content image
        styles (list): List of styles to apply
        frames (int): Number of frames per style
        
    Returns:
        list: List of frames as ASCII art
    """
    if styles is None:
        styles = ["dots", "sketch", "geometric", "blocks"]
    
    frames_list = []
    
    # For each style
    for style in styles:
        # Create multiple frames with increasing style strength
        for i in range(frames):
            strength = (i + 1) / frames
            styled = mix_styles(content_image, style, strength)
            frames_list.append((f"Style: {style}, Strength: {strength:.2f}", styled))
    
    return frames_list

def save_ascii_art(image, filename):
    """Save ASCII art to a file."""
    with open(filename, 'w') as f:
        for row in image:
            f.write(row + '\n')
    print(f"Saved to {filename}")

def main():
    """Main function that runs the ASCII art style transfer demo."""
    # Create a directory for outputs if it doesn't exist
    output_dir = "ascii_art_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate or load content image
    print("Creating content image (house)...")
    content_image = create_house_ascii()
    print("\nContent Image:")
    display_ascii_art(content_image)
    
    # Available styles
    available_styles = ["default", "geometric", "sketch", "dots", "blocks"]
    
    # Apply style transfer with different styles
    for i, style in enumerate(available_styles):
        print(f"\nApplying '{style}' style...")
        
        # Apply the style with 70% strength
        styled_image = mix_styles(content_image, style, 0.7)
        
        # Display the result
        print(f"\nResult with '{style}' style:")
        display_ascii_art(styled_image)
        
        # Save the result
        filename = os.path.join(output_dir, f"styled_{style}.txt")
        save_ascii_art(styled_image, filename)
    
    # Create an animation demonstration
    print("\nCreating style transfer animation frames...")
    animation_frames = simple_animation(content_image)
    
    # Save animation frames
    for i, (frame_title, frame) in enumerate(animation_frames):
        filename = os.path.join(output_dir, f"animation_frame_{i:02d}.txt")
        save_ascii_art(frame, filename)
        print(f"Frame {i}: {frame_title}")
    
    print(f"\nAll images saved to the '{output_dir}' directory")
    
    # Display final comparison
    print("\nFinal comparison:")
    print("\nOriginal content:")
    display_ascii_art(content_image)
    
    print("\nFinal styled version (geometric style):")
    final_style = mix_styles(content_image, "geometric", 0.7)
    display_ascii_art(final_style)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Provide a fallback demo
        print("\nRunning simplified demo instead...")
        
        # Simple house
        house = [
            "    /\\    ",
            "   /  \\   ",
            "  /____\\  ",
            "  |    |  ",
            "  |____|  "
        ]
        
        print("\nOriginal house:")
        for line in house:
            print(line)
        
        # Simple styled version
        styled = []
        for line in house:
            new_line = ""
            for char in line:
                if char != " ":
                    new_line += random.choice(".,-~:;=!*#$@")
                else:
                    new_line += " "
            styled.append(new_line)
        
        print("\nStyled house:")
        for line in styled:
            print(line)
