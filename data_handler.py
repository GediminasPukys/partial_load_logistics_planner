import pandas as pd
import streamlit as st
from datetime import datetime
import io

def load_data(file):
    """Load and preprocess the input data file"""
    try:
        # Handle both file objects and filenames
        if isinstance(file, str):
            # Example data or saved file
            df = pd.read_csv(file)
        else:
            # Uploaded file
            file_data = file.read()
            df = pd.read_csv(io.BytesIO(file_data))
        
        # Convert time strings to datetime objects
        df['Available_from'] = pd.to_datetime(df['Available_from'])
        df['Available_to'] = pd.to_datetime(df['Available_to'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def validate_data(df):
    """Validate the input data for required columns and data types"""
    required_columns = ['id', 'Available_from', 'Available_to', 'quantity', 'Latitude', 'Longitude']
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return False
    
    # Check for data issues
    if df['quantity'].min() <= 0:
        st.warning("Some orders have zero or negative quantity. These orders may be skipped during planning.")
    
    # Check time windows for logical consistency
    invalid_windows = df[df['Available_from'] >= df['Available_to']]
    if not invalid_windows.empty:
        st.error(f"Found {len(invalid_windows)} orders with invalid time windows (from >= to). These orders will be excluded.")
        st.dataframe(invalid_windows)
        # Don't return False here to allow processing with warnings
    
    # Check for missing/null values
    null_counts = df[required_columns].isnull().sum()
    null_columns = null_counts[null_counts > 0].index.tolist()
    if null_columns:
        st.error(f"Found null values in columns: {', '.join(null_columns)}")
        return False
    
    # Check for valid coordinates
    if (df['Latitude'].min() < -90 or df['Latitude'].max() > 90 or 
        df['Longitude'].min() < -180 or df['Longitude'].max() > 180):
        st.error("Found invalid coordinates outside the valid range.")
        return False
    
    return True

def create_example_data():
    """Create an example data file"""
    data = [
        {"id": 1, "Available_from": "2025-03-18 07:00:00", "Available_to": "2025-03-18 10:00:00", "quantity": 3.25, "Latitude": 55.689774, "Longitude": 12.519797},
        {"id": 2, "Available_from": "2025-03-18 07:00:00", "Available_to": "2025-03-18 10:00:00", "quantity": 2.5, "Latitude": 55.697546, "Longitude": 12.585792},
        {"id": 3, "Available_from": "2025-03-18 05:00:00", "Available_to": "2025-03-18 08:00:00", "quantity": 11, "Latitude": 55.748816, "Longitude": 12.320191},
        {"id": 4, "Available_from": "2025-03-18 08:00:00", "Available_to": "2025-03-18 16:00:00", "quantity": 5.75, "Latitude": 55.663388, "Longitude": 11.410811},
        {"id": 5, "Available_from": "2025-03-18 10:00:00", "Available_to": "2025-03-18 13:00:00", "quantity": 5.5, "Latitude": 55.68266, "Longitude": 12.577234}
    ]
    df = pd.DataFrame(data)
    df.to_csv("example_data.csv", index=False)
    return "example_data.csv"
