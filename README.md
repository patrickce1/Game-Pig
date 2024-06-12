# Game Pig
<img width="1378" alt="image" src="https://github.com/patrickce1/Game-Pig/assets/157386969/4821d41e-d217-4c3e-95a0-b946ccc87d0e">

<img width="1378" alt="image" src="https://github.com/patrickce1/Game-Pig/assets/157386969/18a813eb-70b1-4e84-906c-6921f6fc51e3">


## Contents

- [Summary](#summary)
- [Running Locally](#running-locally)
- [Debugging Some Basic Errors](#debugging-some-basic-errors)
- [Virtual Environments and Dependency Tracking](#virtual-environments-and-dependency-tracking)
- [Troubleshooting](#troubleshooting)


## Summary

Please view a live demonstration of the app here: https://drive.google.com/file/d/1dEoQwssMMbEQnyHvLB0z5Q33XIGZElcY/view?usp=sharing

## Running locally

- This is not formally a requirement of P01.  This is to help you test and develop your app locally; we recommend each member of the team to try this out. 
- Ensure that you have Python version 3.10 or above installed on your machine (ideally in a virtual environment). Some of the libraries and code used in the template, as well as on the server end, are only compatible with Python versions 3.10 and above.
  
### Step 1: Set up a virtual environment
Create a virtual environment in Python. You may continue using the one you setup for assignment if necessary. To review how to set up a virtual environment and activate it, refer to A0 assignment writeup.

Run `python -m venv <virtual_env_name>` in your project directory to create a new virtual environment, remember to change <virtual_env_name> to your preferred environment name.

### Step 2: Install dependencies
You need to install dependencies by running `python -m pip install -r requirements.txt` in the backend folder.

### Step 3: Modify init.json file
This project gives you an init.json file with dummy data to see how app.py file reads data from the json file. 
You can change data in this file to your project's json data, but do not delete or change the name of the file. However, you are allowed to create more json files for your project. 

## Command to run project locally: 
```flask run --host=0.0.0.0 --port=5000```


## Debugging Some Basic Errors
- After the build, wait a few seconds as the server will still be loading, especially for larger applications with a lot of setup
- **Do not change the Dockerfiles without permission**
- Sometimes, if a deployment doesn't work, you can try logging out and back in to see if it works
- Alternatively, checking the console will tell you what error it is. If it's a 401, then logging in and out should fix it. 
- If it isn't a 401, first try checking the logs or container status. Check if the containers are alive or not, which could cause issues. If the containers are down, try stopping and starting them. If that does not work, you can report it on ED.
- If data isn't important, destroying and then cloning and re-building containers will usually fix the issue (assuming there's no logical error)

## Virtual Environments and Dependency Tracking
- It's essential to avoid uploading your virtual environments, as they can significantly inflate the size of your project. Large repositories will lead to issues during cloning, especially when memory limits are crossed (Limit – 2GB). 
To prevent your virtual environment from being tracked and uploaded to GitHub, follow these steps:
1. **Exclude Virtual Environment**
   - Navigate to your project's root directory and locate the `.gitignore` file. 
   - Add the name of your virtual environment directory to this file in the following format: `<virtual_environment_name>/`. This step ensures that Git ignores the virtual environment folder during commits.

2. **Remove Previously Committed Virtual Environment**
   - If you've already committed your virtual environment to the repository, you can remove it from the remote repository by using Git commands to untrack and delete it. You will find resources online to do so.
Afterward, ensure to follow step 1 to prevent future tracking of virtual environment.

3. **Managing Dependencies**
    - Add all the new libraries you downloaded using pip install for your project to the existing `requirements.txt` file. To do so,
    - Navigate to your project backend directory and run the command `pip freeze > requirements.txt`. This command will create or overwrite the `requirements.txt` file with a list of installed packages and their versions. 
    - Our server will use your project’s `requirements.txt` file to install all required packages, ensuring that your project runs seamlessly.

