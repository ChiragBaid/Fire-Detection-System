import os

def rename_images(directory, start_number):
    """
    Rename image files in the specified directory to fire_XXXX format.

    Args:
        directory (str): Path to the directory containing the images.
        start_number (int): The starting number for renaming.
    """
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    # Get a list of files in the directory
    files = os.listdir(directory)

    # Filter only image files (you can customize the extensions)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    images = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]

    # Sort images to ensure consistent renaming
    images.sort()

    # Rename images
    for index, image in enumerate(images, start=start_number):
        # Generate the new filename
        new_name = f"no_fire_{index:05}.jpg"  # Change ".jpg" to the desired extension if needed

        # Get the full paths
        old_path = os.path.join(directory, image)
        new_path = os.path.join(directory, new_name)

        # Rename the file
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {image} -> {new_name}")
        except Exception as e:
            print(f"Error renaming {image}: {e}")

if __name__ == "__main__":
    # Specify the directory containing images
    image_directory = r"C:\Users\Admin\Desktop\Fire Detection From a Video\Datasets\Dataset\Training and Validation\nofire"

    # Specify the starting number for the filenames
    starting_number = 1

    # Call the function
    rename_images(image_directory, starting_number)
