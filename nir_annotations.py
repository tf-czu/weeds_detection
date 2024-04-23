import os
import sys
import xml.etree.ElementTree as ET
from shutil import copyfile

def update_xml_file(xml_path, new_name):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Update filename in the XML
    filename_element = root.find('filename')
    filename_element.text = new_name

    # Update file extension in the XML
    filename_element.text = filename_element.text.replace('.xml', '.tiff')

    # Save the modified XML
    new_xml_path = xml_path.replace("rgb", "nir")
    tree.write(new_xml_path)

def process_directory(directory_path):
    # Iterate through all XML files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".xml") and "rgb" in filename:
            xml_path = os.path.join(directory_path, filename)

            # Create a copy of the XML file with updated name and content
            new_name = filename.replace("rgb", "nir")
            update_xml_file(xml_path, new_name)

if __name__ == "__main__":
    # Check if the directory path is provided as a command line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py /path/to/your/directory")
        sys.exit(1)

    # Get the directory path from the command line argument
    directory_path = sys.argv[1]

    # Process the directory
    process_directory(directory_path)
