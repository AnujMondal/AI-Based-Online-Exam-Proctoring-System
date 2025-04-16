#!/usr/bin/env python3
"""
Demo Simulation for Enhanced Proctoring System
This script demonstrates various proctoring events through simulations.
"""
import os
import time
import requests
import random
import sys
import webbrowser
import json

# Configuration
BASE_URL = "http://127.0.0.1:5004"  # Updated to correct port shown in logs
API_ENDPOINTS = {
    "toggle_phone": "/api/toggle-simulate-phone",
    "toggle_book": "/api/toggle-simulate-book",
    "disable_all": "/api/disable-all-simulations",
    "status": "/api/simulation-status"
}

def print_header(text):
    """Display formatted header text"""
    print("\n" + "=" * 50)
    print(f"    {text}")
    print("=" * 50)

def print_status(response):
    """Print simulation status from API response"""
    if response.status_code == 200:
        data = response.json()
        if data.get("success"):
            print(f"Phone simulation: {'ENABLED' if data.get('phone_simulation') else 'DISABLED'}")
            print(f"Book simulation: {'ENABLED' if data.get('book_simulation') else 'DISABLED'}")
        else:
            print(f"Error: {data.get('error', 'Unknown error')}")
    else:
        print(f"HTTP Error: {response.status_code}")

def run_demo():
    """Run the proctoring system simulation demo"""
    print_header("ENHANCED PROCTORING SYSTEM DEMO")
    print("This demo will simulate various proctoring events.")
    print("Make sure the proctoring server is running!")
    
    # First check if the server is running
    try:
        response = requests.get(f"{BASE_URL}/api/simulation-status", timeout=5)
        if response.status_code != 200:
            print("Error: Proctoring server is not responding correctly")
            return
    except requests.exceptions.RequestException:
        print(f"Error: Could not connect to proctoring server at {BASE_URL}")
        print("Make sure the Flask app is running (python app.py)")
        return
    
    # Disable all simulations at start
    print("\nResetting all simulations...")
    response = requests.post(f"{BASE_URL}/api/disable-all-simulations")
    print_status(response)
    
    # Open the exam page in browser
    print("\nOpening exam page in browser...")
    webbrowser.open(f"{BASE_URL}/start-test")
    time.sleep(2)
    
    # Simulation sequence
    print("\nStarting simulation sequence in 5 seconds...")
    for i in range(5, 0, -1):
        print(f"{i}...", end=" ", flush=True)
        time.sleep(1)
    print("\n")
    
    # Simulate phone detection
    print_header("SIMULATING PHONE DETECTION")
    print("Enabling phone simulation for 10 seconds...")
    response = requests.post(f"{BASE_URL}/api/toggle-simulate-phone")
    print_status(response)
    
    print("\nThe system should now show a phone detection alert")
    print("⚠️ You should see a warning in the exam interface")
    time.sleep(10)
    
    # Disable phone simulation
    print("\nDisabling phone simulation...")
    response = requests.post(f"{BASE_URL}/api/toggle-simulate-phone")
    print_status(response)
    time.sleep(2)
    
    # Simulate book detection
    print_header("SIMULATING BOOK DETECTION")
    print("Enabling book simulation for 10 seconds...")
    response = requests.post(f"{BASE_URL}/api/toggle-simulate-book")
    print_status(response)
    
    print("\nThe system should now show a book detection alert")
    print("⚠️ You should see a warning in the exam interface")
    time.sleep(10)
    
    # Disable book simulation
    print("\nDisabling book simulation...")
    response = requests.post(f"{BASE_URL}/api/toggle-simulate-book")
    print_status(response)
    time.sleep(2)
    
    # Simulate both phone and book at the same time
    print_header("SIMULATING MULTIPLE PROHIBITED ITEMS")
    print("Enabling both phone and book simulation for 10 seconds...")
    requests.post(f"{BASE_URL}/api/toggle-simulate-phone")
    requests.post(f"{BASE_URL}/api/toggle-simulate-book")
    response = requests.get(f"{BASE_URL}/api/simulation-status")
    print_status(response)
    
    print("\nThe system should now show multiple prohibited items")
    print("⚠️ You should see warnings for both items in the exam interface")
    time.sleep(10)
    
    # Final cleanup - disable all simulations
    print_header("DEMO COMPLETE")
    print("Cleaning up - disabling all simulations...")
    response = requests.post(f"{BASE_URL}/api/disable-all-simulations")
    print_status(response)
    
    print("\nDemo simulation complete. You can now continue using the system normally.")
    print("To run this demo again, execute: python demo_simulation.py")

if __name__ == "__main__":
    run_demo() 