import os
from torchvision import transforms
from PIL import Image


def grayscale_images():
    # Define the path to your images
    input_folder = 'data/MGS_data/data'
    output_folder = 'data/MGS_data/grayscaled_data'

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define the transformation
    transform = transforms.Grayscale(num_output_channels=1)

    def grayscaling_images(input_folder, output_folder):
        """Process images by converting to grayscale."""
        for filename in os.listdir(input_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG')):
                img_path = os.path.join(input_folder, filename)
                image = Image.open(img_path)

                # Apply the transformation and save
                gray_image = transform(image)
                gray_image.save(os.path.join(output_folder, filename))
                print(f"Processed and saved {filename}.")

    # Run the processing function
    grayscaling_images(input_folder, output_folder)