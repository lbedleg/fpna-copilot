# Lucas Bedleg
# Email: lulu.bm9000@gmail.com
# CFO Copilot Project
# Date: September 26th 2025
# File: __init__.py
# Description: Initializes the CFO Copilot package by exposing core functions for intent classification, data loading, 
# and answering finance-related queries.

from .classifier import classify_intent
from .tools import load_data, answer
