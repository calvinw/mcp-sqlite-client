{ pkgs, ... }: {
  # Define needed packages
  packages = [
    pkgs.python3
    pkgs.python3Packages.virtualenv
    # Add other packages you need
  ];
  
  # Environment variables if needed
  env = { 
    # Any environment variables your project needs
  };
  
  # Workspace lifecycle hooks
  idx.workspace = {
    # Commands that run when workspace starts
    onStart = {
      # This activates your virtualenv - note we prefix with "source "
      activate-venv = "source venv/bin/activate";
      
      # You can add other startup commands here
      # example = "echo 'Workspace started'";
    };
  };
}