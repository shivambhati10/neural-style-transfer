import os
import math
import random
from collections import defaultdict

class SimpleNeuralStyleTransfer:
    """
    A simplified simulation of neural style transfer without external dependencies.
    
    This class provides a pedagogical implementation that demonstrates the concepts
    behind neural style transfer using only standard Python libraries.
    """
    
    def __init__(self, width=256, height=256):
        self.width = width
        self.height = height
        self.content_features = {}
        self.style_features = {}
        self.output_image = []
        
    def load_ascii_image(self, filepath):
        """
        Load an ASCII art image from a file.
        
        Args:
            filepath (str): Path to the ASCII art file
            
        Returns:
            list: 2D representation of the ASCII image
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Remove newlines and ensure consistent width
            image = [line.rstrip('\n') for line in lines]
            max_width = max(len(line) for line in image)
            
            # Pad lines to ensure consistent width
            image = [line.ljust(max_width) for line in image]
            
            return image
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            # Return a small default image
            return ['.....', '.....', '.....', '.....', '.....']
    
    def create_ascii_image(self, chars=None, width=None, height=None):
        """
        Create a random ASCII art image.
        
        Args:
            chars (str, optional): Characters to use in the image
            width (int, optional): Width of the image
            height (int, optional): Height of the image
            
        Returns:
            list: 2D representation of the ASCII image
        """
        if chars is None:
            chars = ".,-~:;=!*#$@"
        
        width = width or self.width
        height = height or self.height
        
        image = []
        for _ in range(height):
            row = ''.join(random.choice(chars) for _ in range(width))
            image.append(row)
        
        return image
    
    def extract_content_features(self, image):
        """
        Extract 'content features' from an ASCII image.
        
        In real neural style transfer, these would be activations from 
        higher layers of a CNN. Here we use statistical properties of the ASCII image.
        
        Args:
            image (list): 2D ASCII image
            
        Returns:
            dict: Extracted content features
        """
        features = {}
        
        # Extract character frequency
        char_freq = defaultdict(int)
        for row in image:
            for char in row:
                char_freq[char] += 1
        
        # Normalize frequencies
        total_chars = sum(char_freq.values())
        for char, count in char_freq.items():
            char_freq[char] = count / total_chars
        
        features['char_distribution'] = dict(char_freq)
        
        # Extract line length variance
        line_lengths = [len(line) for line in image]
        avg_length = sum(line_lengths) / len(line_lengths)
        variance = sum((length - avg_length) ** 2 for length in line_lengths) / len(line_lengths)
        
        features['line_length_variance'] = variance
        
        # Extract "edges" - transitions between different characters
        edges = 0
        for row in image:
            for i in range(1, len(row)):
                if row[i] != row[i-1]:
                    edges += 1
        
        features['edge_density'] = edges / (sum(len(row) for row in image))
        
        return features
    
    def extract_style_features(self, image):
        """
        Extract 'style features' from an ASCII image.
        
        In real neural style transfer, these would be Gram matrices from 
        different layers of a CNN. Here we use pattern statistics from the ASCII image.
        
        Args:
            image (list): 2D ASCII image
            
        Returns:
            dict: Extracted style features
        """
        features = {}
        
        # Extract character transition probabilities (like a Gram matrix)
        transition_matrix = defaultdict(lambda: defaultdict(int))
        
        # Horizontal transitions
        for row in image:
            for i in range(len(row) - 1):
                transition_matrix[row[i]][row[i+1]] += 1
        
        # Normalize transitions
        for char, transitions in transition_matrix.items():
            total = sum(transitions.values())
            if total > 0:
                for next_char, count in transitions.items():
                    transition_matrix[char][next_char] = count / total
        
        features['transition_matrix'] = {k: dict(v) for k, v in transition_matrix.items()}
        
        # Extract character density patterns
        density_patterns = []
        for i in range(0, len(image) - 1, 2):
            for j in range(0, len(image[0]) - 1, 2):
                # Get 2x2 patch
                patch = [row[j:j+2] for row in image[i:i+2]]
                flat_patch = ''.join([''.join(row) for row in patch])
                density_patterns.append(flat_patch)
        
        pattern_counts = defaultdict(int)
        for pattern in density_patterns:
            pattern_counts[pattern] += 1
        
        # Normalize pattern counts
        total_patterns = sum(pattern_counts.values())
        if total_patterns > 0:
            for pattern, count in pattern_counts.items():
                pattern_counts[pattern] = count / total_patterns
        
        features['density_patterns'] = dict(pattern_counts)
        
        return features
    
    def compute_content_loss(self, target_features, current_features):
        """
        Compute content loss between target and current features.
        
        Args:
            target_features (dict): Target content features
            current_features (dict): Current content features
            
        Returns:
            float: Content loss
        """
        # Compare character distributions
        dist_loss = 0
        for char, freq in target_features['char_distribution'].items():
            current_freq = current_features['char_distribution'].get(char, 0)
            dist_loss += (freq - current_freq) ** 2
        
        # Compare edge density
        edge_loss = (target_features['edge_density'] - current_features['edge_density']) ** 2
        
        # Compare line length variance
        var_loss = (target_features['line_length_variance'] - current_features['line_length_variance']) ** 2
        
        # Weighted sum of losses
        total_loss = dist_loss + 2 * edge_loss + var_loss
        
        return total_loss
    
    def compute_style_loss(self, target_features, current_features):
        """
        Compute style loss between target and current features.
        
        Args:
            target_features (dict): Target style features
            current_features (dict): Current style features
            
        Returns:
            float: Style loss
        """
        # Compare transition matrices
        trans_loss = 0
        target_trans = target_features['transition_matrix']
        current_trans = current_features['transition_matrix']
        
        for char, transitions in target_trans.items():
            for next_char, prob in transitions.items():
                current_prob = current_trans.get(char, {}).get(next_char, 0)
                trans_loss += (prob - current_prob) ** 2
        
        # Compare density patterns
        pattern_loss = 0
        target_patterns = target_features['density_patterns']
        current_patterns = current_features['density_patterns']
        
        for pattern, freq in target_patterns.items():
            current_freq = current_patterns.get(pattern, 0)
            pattern_loss += (freq - current_freq) ** 2
        
        # Weighted sum of losses
        total_loss = trans_loss + 2 * pattern_loss
        
        return total_loss
    
    def modify_image(self, image, num_changes=10):
        """
        Make random modifications to the image.
        
        Args:
            image (list): 2D ASCII image
            num_changes (int): Number of modifications to make
            
        Returns:
            list: Modified 2D ASCII image
        """
        # Create a deep copy of the image
        new_image = [row for row in image]
        
        for _ in range(num_changes):
            i = random.randint(0, len(new_image) - 1)
            j = random.randint(0, len(new_image[0]) - 1)
            
            # Get possible characters from style image
            possible_chars = list(self.style_features['char_distribution'].keys())
            if not possible_chars:
                possible_chars = ['.', '-', '~', '*', '#', '@']
            
            # Choose a new character weighted by style distribution
            weights = [self.style_features['char_distribution'].get(char, 0.1) for char in possible_chars]
            total_weight = sum(weights)
            if total_weight == 0:
                weights = [1/len(possible_chars)] * len(possible_chars)
            else:
                weights = [w/total_weight for w in weights]
            
            new_char = random.choices(possible_chars, weights=weights, k=1)[0]
            
            # Modify the character
            row = list(new_image[i])
            row[j] = new_char
            new_image[i] = ''.join(row)
        
        return new_image
    
    def run_style_transfer(self, content_path=None, style_path=None, iterations=1000, content_weight=0.5):
        """
        Run the simplified style transfer algorithm.
        
        Args:
            content_path (str, optional): Path to content ASCII art file
            style_path (str, optional): Path to style ASCII art file
            iterations (int): Number of iterations to run
            content_weight (float): Weight of content loss vs style loss (0-1)
            
        Returns:
            list: The resulting stylized ASCII image
        """
        # Load content and style images, or use defaults
        if content_path and os.path.exists(content_path):
            content_image = self.load_ascii_image(content_path)
        else:
            print("Using default content image")
            content_image = self.create_ascii_image(chars=".,-~:;=!*#$@")
        
        if style_path and os.path.exists(style_path):
            style_image = self.load_ascii_image(style_path)
        else:
            print("Using default style image")
            style_image = self.create_ascii_image(chars="01")
        
        # Resize content image to match style image dimensions
        height = len(style_image)
        width = len(style_image[0])
        
        content_image = content_image[:height]
        content_image = [row[:width] for row in content_image]
        
        # Pad if needed
        while len(content_image) < height:
            content_image.append(' ' * width)
        
        content_image = [row.ljust(width) for row in content_image]
        
        # Extract features
        self.content_features = self.extract_content_features(content_image)
        self.style_features = self.extract_style_features(style_image)
        
        # Initialize with content image
        current_image = [row for row in content_image]
        best_image = current_image
        best_loss = float('inf')
        
        style_weight = 1.0 - content_weight
        
        print("Starting style transfer...")
        
        for i in range(iterations):
            # Modify the image
            new_image = self.modify_image(current_image)
            
            # Extract features from the new image
            new_features_content = self.extract_content_features(new_image)
            new_features_style = self.extract_style_features(new_image)
            
            # Compute losses
            content_loss = self.compute_content_loss(self.content_features, new_features_content)
            style_loss = self.compute_style_loss(self.style_features, new_features_style)
            
            # Total loss with weighting
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            # Decide whether to accept the new image (simulated annealing)
            temperature = 1.0 - (i / iterations)  # Starts at 1, ends at 0
            acceptance_probability = math.exp(-max(0, total_loss - best_loss) / (temperature + 1e-10))
            
            if total_loss < best_loss or random.random() < acceptance_probability:
                current_image = new_image
                
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_image = new_image
            
            # Print progress
            if (i + 1) % (iterations // 10) == 0:
                progress = (i + 1) / iterations * 100
                print(f"Progress: {progress:.1f}% - Loss: {total_loss:.4f}")
        
        self.output_image = best_image
        return best_image
    
    def save_image(self, image, filepath):
        """
        Save an ASCII image to a file.
        
        Args:
            image (list): 2D ASCII image
            filepath (str): Path to save the file
        """
        with open(filepath, 'w') as f:
            for row in image:
                f.write(row + '\n')
        
        print(f"Image saved to {filepath}")
    
    def display_image(self, image):
        """
        Display an ASCII image.
        
        Args:
            image (list): 2D ASCII image
        """
        border = '+' + '-' * len(image[0]) + '+'
        print(border)
        for row in image:
            print(f"|{row}|")
        print(border)
    
    def compare_images(self, content_image, style_image, result_image):
        """
        Display content, style, and result images side by side.
        
        Args:
            content_image (list): Content ASCII image
            style_image (list): Style ASCII image
            result_image (list): Result ASCII image
        """
        # Normalize widths for display
        max_width = max(
            max(len(row) for row in content_image),
            max(len(row) for row in style_image),
            max(len(row) for row in result_image)
        )
        
        content_norm = [row.ljust(max_width) for row in content_image]
        style_norm = [row.ljust(max_width) for row in style_image]
        result_norm = [row.ljust(max_width) for row in result_image]
        
        # Ensure all have the same number of rows
        max_rows = max(len(content_norm), len(style_norm), len(result_norm))
        
        content_norm = content_norm + [' ' * max_width] * (max_rows - len(content_norm))
        style_norm = style_norm + [' ' * max_width] * (max_rows - len(style_norm))
        result_norm = result_norm + [' ' * max_width] * (max_rows - len(result_norm))
        
        # Print headers
        spacing = 4
        total_width = max_width * 3 + spacing * 2
        
        print('=' * total_width)
        header = 'CONTENT'.center(max_width) + ' ' * spacing + 'STYLE'.center(max_width) + ' ' * spacing + 'RESULT'.center(max_width)
        print(header)
        print('=' * total_width)
        
        # Print rows
        for i in range(max_rows):
            row = content_norm[i] + ' ' * spacing + style_norm[i] + ' ' * spacing + result_norm[i]
            print(row)
        
        print('=' * total_width)

def create_sample_ascii_art(sample_name, output_file):
    """
    Create a simple ASCII art sample file.
    
    Args:
        sample_name (str): Name of the sample ("content" or "style")
        output_file (str): Path to save the file
    """
    if sample_name == "content":
        # Simple house
        art = [
            "    /\\    ",
            "   /  \\   ",
            "  /____\\  ",
            "  |    |  ",
            "  |____|  "
        ]
    elif sample_name == "style":
        # Pattern
        art = [
            "+-+-+-+-+-+",
            "|#|#|#|#|#|",
            "+-+-+-+-+-+",
            "|#|#|#|#|#|",
            "+-+-+-+-+-+"
        ]
    else:
        # Default pattern
        art = [
            "...........",
            "...........",
            "...........",
            "...........",
            "..........."
        ]
    
    with open(output_file, 'w') as f:
        for line in art:
            f.write(line + '\n')
    
    print(f"Created sample ASCII art: {output_file}")

def main():
    """Main function to demonstrate the SimpleNeuralStyleTransfer class."""
    # Check if sample files exist, create them if not
    content_file = "content_ascii.txt"
    style_file = "style_ascii.txt"
    
    if not os.path.exists(content_file):
        create_sample_ascii_art("content", content_file)
    
    if not os.path.exists(style_file):
        create_sample_ascii_art("style", style_file)
    
    # Create a style transfer object
    nst = SimpleNeuralStyleTransfer(width=20, height=10)
    
    # Run style transfer
    print("\nRunning ASCII Neural Style Transfer...")
    content_image = nst.load_ascii_image(content_file)
    style_image = nst.load_ascii_image(style_file)
    
    print("\nContent Image:")
    nst.display_image(content_image)
    
    print("\nStyle Image:")
    nst.display_image(style_image)
    
    print("\nApplying style transfer (this may take a moment)...")
    result_image = nst.run_style_transfer(
        content_path=content_file,
        style_path=style_file,
        iterations=500,
        content_weight=0.7
    )
    
    print("\nResult Image:")
    nst.display_image(result_image)
    
    # Compare all images
    print("\nComparison of all images:")
    nst.compare_images(content_image, style_image, result_image)
    
    # Save result
    result_file = "result_ascii.txt"
    nst.save_image(result_image, result_file)

if __name__ == "__main__":
    main()
