#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn api.app:app --reload
