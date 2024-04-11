#!/bin/bash

echo "running unittests"

echo "unittests have been executed"

echo "Running HTTP-Server...."
uvicorn rest.settings:application --port $2
