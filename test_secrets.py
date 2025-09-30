#!/usr/bin/env python3
"""
Test script to verify Streamlit secrets are working correctly
"""
import streamlit as st

def test_secrets():
    """Test if secrets are accessible"""
    try:
        # This should work when run within Streamlit context
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if api_key:
            print(f"✅ API Key found: {api_key[:20]}...")
            return True
        else:
            print("❌ API Key not found in secrets")
            return False
    except Exception as e:
        print(f"❌ Error accessing secrets: {e}")
        return False

if __name__ == "__main__":
    print("Testing Streamlit secrets...")
    test_secrets()
