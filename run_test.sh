#!/bin/bash
coverage run -m pytest -v
coverage report -i
coverage html -i
