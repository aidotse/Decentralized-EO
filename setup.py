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


if __name__ == "__main__":

    # Create directory for external libraries
    if not os.path.exists('modules'):
        os.makedirs('modules')

    # Clone Paseos:
    url = 'https://github.com/aidotse/PASEOS.git'
    path = 'modules/PASEOS'
    branch = 'student'
    clone_repository(url, path, branch)

    # Update current conda environment with dependencies required by Paseos
    #env_name = 'eo'
    #environment_file = 'modules/paseos/environment.yml'
    #update_conda_environment(env_name, environment_file)
