#!/bin/bash

# Exit on error
set -e

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --noinput

# Print success message
echo "Build completed successfully!" 