import numpy as np
from math import radians, cos, sin, asin, sqrt
from datetime import datetime, timedelta, time
import pandas as pd

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def travel_time(distance, speed):
    """Calculate travel time in hours given distance (km) and speed (km/h)"""
    return distance / speed

def combine_date_time(date, time_obj):
    """Combine a date and time object into a datetime object"""
    return datetime.combine(date, time_obj)

def format_time(datetime_obj):
    """Format datetime object to string"""
    if datetime_obj is None:
        return "N/A"
    return datetime_obj.strftime('%H:%M')

def format_duration(hours):
    """Format duration in hours to a readable string"""
    if hours is None:
        return "N/A"
    
    total_minutes = int(hours * 60)
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes}m"

def generate_color(index, total, selected=False):
    """Generate a distinct color for visualization based on index"""
    # Generate a color on the HSV color wheel
    hue = index / total * 360
    
    # Convert HSV to RGB (simplified approach)
    h = hue / 60
    c = 255
    x = 255 * (1 - abs((h % 2) - 1))
    
    if h < 1: rgb = [c, x, 0]
    elif h < 2: rgb = [x, c, 0]
    elif h < 3: rgb = [0, c, x]
    elif h < 4: rgb = [0, x, c]
    elif h < 5: rgb = [x, 0, c]
    else: rgb = [c, 0, x]
    
    # Make selected trucks more visible
    if selected:
        # Brighten and saturate the color
        rgb = [min(val * 1.2, 255) for val in rgb]
    
    return [int(val) for val in rgb]

def truck_utilization(route, max_capacity):
    """Calculate truck capacity utilization percentage"""
    if max_capacity <= 0:
        return 0
    return (route['total_load'] / max_capacity) * 100

def calculate_empty_miles(route):
    """Calculate empty miles percentage"""
    if route['total_distance'] <= 0:
        return 0
    
    # Empty miles are those traveled with no cargo (return trips)
    empty_miles = route['empty_distance']
    return (empty_miles / route['total_distance']) * 100
