"""
Module for generating fake thermal simulation results
"""

# %% Import packages
import numpy as np
import os
import shutil


def generate_fake_data(nPas, nElemsTot, directory_path=r".\thermal_data\fake_data", loadType="uniform", Ncooling=10, Tbuild=353.15, Tsubstrate=323.15, Tamb=293.15):
    """
    Function generating fake temperature data assuming material is deposited at temperature Tbuild and cools down to room temperature Tamb linearly in 'Ncooling' number of steps without ever heating again

    Parameters
    ----------
    Tbuild : float, optional
        Temperature of deposition of the material during the process.
        For a FDM (PLA) process its = 353.15, for a metal process its = 2463.060825.
    Tsubstrate : int, optional
        Temperature of the substrate.
        The default is :  323.15.
    Tamb : int, optional
        Temperature of the room.

    Returns
    -------
    nPas files of nElems*nLayers temperatures (one temperature per line)
    """

    # Check if the directory exists
    if os.path.exists(directory_path):
        # Get sorted list of files in the directory
        files_in_directory = sorted(os.listdir(directory_path))

        # Remove excess files if the directory contains more than nPas files
        if len(files_in_directory) > nPas:
            for file_name in files_in_directory[nPas:]:
                file_path = os.path.join(directory_path, file_name)
                os.remove(file_path)
            print(f"Previous extra files have been removed from {directory_path}.")
    else:
        # Create the directory if it doesn't exist
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")

    # Generate temperatures
    if loadType == 'uniform' :
       temperatures = Tbuild * np.ones(nElemsTot)
    if loadType == 'linear' :
       temperatures = np.linspace(Tamb,Tbuild, nElemsTot)

    if loadType == 'poc' :
        Ncooling = max(2, Ncooling) #linspace needs Ncooling > 1
        temperatures = Tamb * np.ones(nElemsTot)
        temperatures[-Ncooling:] = np.linspace(Tamb,Tbuild, Ncooling)

    # Write temperature files
    for i in range(1, nPas+1):
        file_name = f"temp_{str(i).zfill(4)}.txt"
        file_path = os.path.join(directory_path, file_name)

        with open(file_path, 'w') as file:
            if i <= nElemsTot :
                data = temperatures[-i:]
            else :
                data = np.hstack((np.array((i-nElemsTot)*[Tamb]), temperatures[:-(i-nElemsTot)]))
            file.write('\n'.join(map(str, data)))

    print(f"Temperatures have been generated and stored in {directory_path}.")


def rename_files(directory_path):  # add zeros to file names to ensure 001 to 099 numbering
    # Get a list of all files in the directory
    files = sorted(os.listdir(directory_path))

    # Rename each file
    for file_name in files:
        # Extract the number from the file name
        number_str = file_name.split('_')[1].split('.')[0]

        # Generate the new file name with zero padding
        new_file_name = f"temp_{number_str.zfill(4)}.txt"

        # Construct the full paths for the old and new files
        old_file_path = os.path.join(directory_path, file_name)
        new_file_path = os.path.join(directory_path, new_file_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{file_name}' to '{new_file_name}'")


def generate_carpets(nLayers_v, directory_path=r".\thermal_data\carpet-copy"):
    # Get a list of all files in the directory
    files = os.listdir(directory_path)

    # Create the target directory if it doesn't exist
    target_directory = os.path.join(directory_path, "duplicated_series")
    os.makedirs(target_directory, exist_ok=True)

    # Copy the original files to the target directory with renaming
    for file_name in files:
        source_file = os.path.join(directory_path, file_name)
        new_file_name = f"{file_name.split('_')[0]}_{0}_{file_name.split('_')[-1]}"
        target_file = os.path.join(target_directory, new_file_name)
        shutil.copy(source_file, target_file)

    files = os.listdir(target_directory)
    # Duplicate the files nLayers_h-1 times
    for i in range(1, nLayers_v):
        # Get the last file in the series
        last_file = os.path.join(target_directory, files[-1])

        # Copy each file in the series and append the content of the last file to it
        for file_name in files:
            source_file = os.path.join(target_directory, file_name)
            new_file_name = f"{file_name.split('_')[0]}_{i}_{file_name.split('_')[-1]}"
            target_file = os.path.join(target_directory, new_file_name)

            with open(source_file, 'r') as f:
                content = f.read()

            with open(last_file, 'r') as f:
                last_content = f.read()

            with open(target_file, 'w') as f:
                f.write(content + last_content)

        # Update the list of files to include the newly created ones
        files = sorted(os.listdir(target_directory))
    return
