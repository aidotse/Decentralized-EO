# In this script we will provide the necessary steps for executing scripts in the eo repository.

# Import core modules
import os
import git
import subprocess


def clone_repository(url, path, branch):
    """
    Clones a git repository to you local directory
    using gitpython and adds to 'path'.

    Args:
        url (str): HTTPS url to repository on github.
        path (str): Directory for the locally stored clone.
        branch (str): Name of the repo branch to be cloned.
    """
    git.Repo.clone_from(url, path, branch=branch)


def update_conda_environment(env_name, environment_file):
    """
    Updates the current conda environment based on the 
    dependency requirements of external libraries.
    
    The update is done by passing the current conda 
    env and the path to the file containing a list of dependencies.
    """
    try:
        # Construct the command to update the Conda environment
        command = [
            'conda', 'env', 'update',
            '--name', env_name,
            '--file', environment_file,
            '--prune'
        ]

        # Run the command
        subprocess.run(command, check=True)

        print(f"Environment '{env_name}' successfully updated.")
    except subprocess.CalledProcessError as e:
        print(f"Error updating environment '{env_name}': {e}")

def update_file(file_path):
    try:
        # Read the original content
        with open(file_path, 'r') as file:
            content = file.readlines()
        
        # Modify the specific line
        new_content = []
        for line in content:
            if "sigmoid_output: bool = False" in line:
                new_content.append(line.replace("sigmoid_output: bool = False", "sigmoid_output: bool = True"))
            else:
                new_content.append(line)
        
        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.writelines(new_content)
        
        print(f"File '{file_path}' successfully updated.")
    except Exception as e:
        print(f"Error updating file '{file_path}': {e}")

if __name__ == "__main__":
    # Create directory for external libraries
    if not os.path.exists('modules'):
        print("Creating 'modules' directory")
        os.makedirs('modules')
    else:
        print("'modules' directory already exists")

    # Clone PASEOS repository
    paseos_path = 'modules/PASEOS'
    if not os.path.exists(paseos_path):
        print(f"Creating '{paseos_path}' directory")
        os.makedirs(paseos_path)
        url = 'https://github.com/aidotse/PASEOS.git'
        branch = 'student'
        clone_repository(url, paseos_path, branch)
    else:
        print(f"'{paseos_path}' directory already exists")

    # Clone mobile_sam repository
    mobile_sam_path = 'modules/mobile_sam'
    if not os.path.exists(mobile_sam_path):
        print(f"Creating '{mobile_sam_path}' directory")
        os.makedirs(mobile_sam_path)
        url = 'https://github.com/ChaoningZhang/MobileSAM.git'
        branch = 'master'
        clone_repository(url, mobile_sam_path, branch)
    else:
        print(f"'{mobile_sam_path}' directory already exists")

    # Update mask_decoder.py in mobile_sam
    mask_decoder = 'modules/mobile_sam/mobile_sam/modeling/mask_decoder.py'
    update_file(mask_decoder)

    # Update current conda environment with dependencies required by Paseos
    #env_name = 'eo'
    #environment_file = 'modules/paseos/environment.yml'
    #update_conda_environment(env_name, environment_file)
