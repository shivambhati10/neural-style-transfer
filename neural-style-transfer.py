import os
import tensorflow as tf
import numpy as np
import PIL.Image
import time
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to load and preprocess images
def load_and_process_img(path_to_img, max_dim=512):
    img = load_img(path_to_img)
    img = img_to_array(img)
    
    # Large images slow down processing, so resize images
    long_dim = max(img.shape[:-1])
    scale = max_dim / long_dim
    
    # Preserve aspect ratio
    new_height = int(img.shape[0] * scale)
    new_width = int(img.shape[1] * scale)
    
    img = tf.image.resize(img, (new_height, new_width))
    img = img / 255.0  # Normalize to [0,1]
    
    # Add batch dimension
    img = tf.expand_dims(img, axis=0)
    return img

# Function to convert tensors to images
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Function to deprocess images
def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    
    # Convert BGR to RGB
    x = x[:, :, ::-1]
    
    # Clip values to [0, 255]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Content layer used for the content loss
content_layers = ['block5_conv2']

# Style layers used for the style loss
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def vgg_layers(layer_names):
    """ Creates a VGG model that returns a list of intermediate output values."""
    # Load VGG19 model without classifier (include_top=False)
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]
    
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    """Calculate the Gram matrix of an input tensor."""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
        
    def call(self, inputs):
        """Extract style and content features"""
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        
        content_dict = {content_name: value 
                        for content_name, value 
                        in zip(self.content_layers, content_outputs)}
        
        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}
        
        return {'content': content_dict, 'style': style_dict}

def style_content_loss(outputs, style_targets, content_targets, 
                       style_weight=1e-2, content_weight=1e4):
    """Calculate the total loss"""
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) 
                            for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers
    
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) 
                              for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    
    total_loss = style_loss + content_loss
    return total_loss

@tf.function()
def train_step(image, extractor, style_targets, content_targets, 
               style_weight=1e-2, content_weight=1e4, total_variation_weight=30):
    """Training step for the style transfer"""
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets, 
                                  style_weight, content_weight)
        
        # Add total variation loss to reduce high frequency artifacts
        loss += total_variation_weight * tf.image.total_variation(image)
        
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, 0.0, 1.0))
    return loss

def run_style_transfer(content_path, style_path, epochs=10, steps_per_epoch=100,
                      style_weight=1e-2, content_weight=1e4, total_variation_weight=30):
    """Run the style transfer algorithm"""
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)
    
    # Initialize the extractor model
    extractor = StyleContentModel(style_layers, content_layers)
    
    # Get style and content targets
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    
    # Initialize the image to optimize with the content image
    image = tf.Variable(content_image)
    
    # Initialize optimizer
    global opt
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    
    # Store best results
    best_loss = float('inf')
    best_img = None
    
    # For display
    start_time = time.time()
    
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            loss = train_step(image, extractor, style_targets, content_targets,
                             style_weight, content_weight, total_variation_weight)
            
            if loss < best_loss:
                best_loss = loss
                best_img = image.numpy()
                
            if step % 100 == 0:
                print(f"Epoch: {n+1}/{epochs}, Step: {m+1}/{steps_per_epoch}")
                print(f"Total loss: {loss:.4f}, Time: {time.time() - start_time:.4f}s")
    
    return best_img

def display_images(images, titles=None):
    """Display multiple images side by side"""
    if not isinstance(images, list):
        images = [images]
    
    if titles is None:
        titles = [''] * len(images)
    
    plt.figure(figsize=(15, 10))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
    plt.show()

def main(content_path, style_path, result_path='styled_image.jpg'):
    """Main function to run the style transfer"""
    print("Loading and preprocessing images...")
    
    # Perform style transfer
    print("Running style transfer...")
    best_img = run_style_transfer(content_path, style_path)
    
    # Save the result
    result_img = tensor_to_image(best_img)
    result_img.save(result_path)
    print(f"Result saved to {result_path}")
    
    # Display images
    content_img = PIL.Image.open(content_path)
    style_img = PIL.Image.open(style_path)
    
    display_images([content_img, style_img, result_img], 
                  ['Content Image', 'Style Image', 'Result Image'])
    
    return result_img

if __name__ == "__main__":
    # Example usage:
    content_path = "content.jpg"  # Path to your content image
    style_path = "style.jpg"      # Path to your style image
    
    # Check if paths exist and are valid
    if not os.path.exists(content_path):
        print(f"Content image not found at {content_path}")
        exit()
    
    if not os.path.exists(style_path):
        print(f"Style image not found at {style_path}")
        exit()
    
    main(content_path, style_path)
