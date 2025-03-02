#!/bin/bash
flask db upgrade
gunicorn app:app