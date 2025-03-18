import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from utils import haversine_distance, travel_time, combine_date_time

def optimize_routes(df, fleet_params):
    """
    Optimize truck routes based on the given orders and fleet parameters
    
    Args:
        df: DataFrame with order information
        fleet_params: Dictionary with fleet parameters
        
    Returns:
        routes: Dictionary of routes for each truck
        unplanned_orders: List of orders that couldn't be planned
        unused_trucks: Number of unused trucks
    """
    # Extract parameters
    num_trucks = fleet_params['num_trucks']
    warehouse_coords = fleet_params['warehouse_coords']
    max_capacity = fleet_params['max_capacity']
    avg_speed = fleet_params['avg_speed']
    load_unload_time = fleet_params['load_unload_time']
    workday_length = fleet_params['workday_length']
    truck_start_times = fleet_params.get('truck_start_times', {})
    current_date = fleet_params.get('date', datetime.now().date())
    
    # Convert load/unload time to hours
    load_unload_hours = load_unload_time / 60
    
    # Initialize routes for each truck
    routes = {}
    for i in range(1, num_trucks + 1):
        # Use custom start time if available, otherwise default to 8 AM
        start_time = truck_start_times.get(i, datetime.now().time().replace(hour=8, minute=0, second=0, microsecond=0))
        
        # Combine date and time
        start_datetime = combine_date_time(current_date, start_time)
        
        routes[i] = {
            'orders': [],
            'start_time': start_datetime,
            'end_time': None,
            'total_distance': 0,
            'empty_distance': 0,  # Track empty miles separately
            'total_load': 0,
            'waypoints': [],
            'current_load': 0  # Track current load for partial deliveries
        }
    
    # Make a copy of the dataframe to avoid modifying the original
    orders = df.copy()
    
    # Filter out orders with invalid time windows
    valid_orders = orders[orders['Available_from'] < orders['Available_to']]
    if len(valid_orders) < len(orders):
        st.warning(f"Excluded {len(orders) - len(valid_orders)} orders with invalid time windows.")
    
    orders = valid_orders
    
    # Sort orders by several criteria to improve optimization:
    # 1. First by earliest availability window end time (urgent orders first)
    # 2. Then by quantity (larger orders first to ensure they are planned)
    orders = orders.sort_values(['Available_to', 'quantity'], ascending=[True, False])
    
    # Mark all orders as unplanned initially
    unplanned_orders = list(orders['id'])
    
    # Calculate distance matrix for efficiency
    unique_locations = [(row['Latitude'], row['Longitude']) for _, row in orders.iterrows()]
    unique_locations.append(warehouse_coords)  # Add warehouse
    
    # Progress bar for optimization
    progress_bar = st.progress(0)
    n_orders = len(orders)
    
    # For each order, try to assign it to the best truck
    for idx, (_, order) in enumerate(orders.iterrows()):
        # Update progress
        progress_bar.progress((idx + 1) / n_orders)
        
        order_id = order['id']
        order_lat = order['Latitude']
        order_lon = order['Longitude']
        order_quantity = order['quantity']
        order_from = order['Available_from']
        order_to = order['Available_to']
        
        best_truck = None
        min_extra_distance = float('inf')
        best_insertion_position = None
        
        # Try each truck
        for truck_id, truck_route in routes.items():
            # Skip if truck is already full
            if truck_route['current_load'] + order_quantity > max_capacity:
                continue
            
            # Try inserting at different positions in the route
            positions_to_try = range(len(truck_route['waypoints']) + 1)
            
            for position in positions_to_try:
                # Check feasibility and calculate extra distance for this insertion
                feasible, extra_dist, new_schedule = check_insertion_feasibility(
                    truck_route, position, order, warehouse_coords, 
                    avg_speed, load_unload_hours, workday_length
                )
                
                if feasible and extra_dist < min_extra_distance:
                    min_extra_distance = extra_dist
                    best_truck = truck_id
                    best_insertion_position = position
                    best_schedule = new_schedule
        
        # Assign order to the best truck
        if best_truck is not None:
            truck_route = routes[best_truck]
            
            # Insert the order at the best position
            order_copy = order.copy()
            insert_order_at_position(
                truck_route, best_insertion_position, order_copy, 
                warehouse_coords, avg_speed, load_unload_hours, 
                min_extra_distance, best_schedule
            )
            
            # Mark as planned
            if order_id in unplanned_orders:
                unplanned_orders.remove(order_id)
    
    # Hide progress bar
    progress_bar.empty()
    
    # Finalize routes: add return to warehouse if not already added
    for truck_id, truck_route in routes.items():
        if truck_route['orders']:
            finalize_route(truck_route, warehouse_coords, avg_speed)
    
    # Count unused trucks
    unused_trucks = sum(1 for truck_route in routes.values() if not truck_route['orders'])
    
    return routes, unplanned_orders, unused_trucks

def check_insertion_feasibility(truck_route, position, order, warehouse_coords, avg_speed, load_unload_hours, workday_length):
    """
    Check if inserting an order at a specific position in the route is feasible
    and calculate the extra distance needed
    """
    order_lat = order['Latitude']
    order_lon = order['Longitude']
    order_quantity = order['quantity']
    order_from = order['Available_from']
    order_to = order['Available_to']
    
    warehouse_lat, warehouse_lon = warehouse_coords
    
    # Create a temporary copy of the route
    temp_waypoints = truck_route['waypoints'].copy()
    
    # If route is empty, check if we can go from warehouse to order and back
    if not temp_waypoints:
        # Calculate distances
        dist_to_order = haversine_distance(warehouse_lat, warehouse_lon, order_lat, order_lon)
        dist_to_warehouse = haversine_distance(order_lat, order_lon, warehouse_lat, warehouse_lon)
        total_distance = dist_to_order + dist_to_warehouse
        
        # Calculate times
        start_time = truck_route['start_time']
        travel_to_order = travel_time(dist_to_order, avg_speed)
        arrival_time = start_time + timedelta(hours=travel_to_order)
        
        # Check if we arrive within the time window
        if arrival_time > order_to:
            return False, 0, None
        
        # If we arrive before the time window starts, wait
        if arrival_time < order_from:
            arrival_time = order_from
        
        # Service time
        service_end = arrival_time + timedelta(hours=load_unload_hours)
        
        # Return to warehouse
        return_travel = travel_time(dist_to_warehouse, avg_speed)
        return_time = service_end + timedelta(hours=return_travel)
        
        # Check if total work time is within limits
        work_hours = (return_time - start_time).total_seconds() / 3600
        if work_hours > workday_length:
            return False, 0, None
        
        # Feasible insertion
        new_schedule = {
            'arrival': arrival_time,
            'service_end': service_end,
            'return_time': return_time
        }
        
        return True, total_distance, new_schedule
    
    # If inserting in an existing route, need to recalculate the entire schedule
    new_waypoints = temp_waypoints.copy()
    
    # Create the new waypoint
    new_waypoint = {
        'order_id': order['id'],
        'coords': (order_lat, order_lon),
        'arrival_time': None,  # Will calculate
        'end_time': None,      # Will calculate
        'quantity': order_quantity,
        'distance_from_prev': 0,  # Will calculate
        'distance_to_next': 0     # Will calculate
    }
    
    # Insert the new waypoint
    new_waypoints.insert(position, new_waypoint)
    
    # Recalculate distances and times
    current_time = truck_route['start_time']
    current_pos = warehouse_coords
    total_distance = 0
    
    for i, wp in enumerate(new_waypoints):
        wp_lat, wp_lon = wp['coords']
        
        # Calculate distance from previous position
        distance = haversine_distance(current_pos[0], current_pos[1], wp_lat, wp_lon)
        wp['distance_from_prev'] = distance
        total_distance += distance
        
        # Calculate arrival time
        travel_hours = travel_time(distance, avg_speed)
        arrival_time = current_time + timedelta(hours=travel_hours)
        
        # Check if we arrive within the time window
        # For the inserted order, check against its time window
        if i == position:
            if arrival_time > order_to:
                return False, 0, None
            
            # If we arrive before the time window starts, wait
            if arrival_time < order_from:
                arrival_time = order_from
        
        wp['arrival_time'] = arrival_time
        
        # Service time
        service_end = arrival_time + timedelta(hours=load_unload_hours)
        wp['end_time'] = service_end
        
        # Update for next waypoint
        current_time = service_end
        current_pos = (wp_lat, wp_lon)
    
    # Add return to warehouse
    distance_to_warehouse = haversine_distance(current_pos[0], current_pos[1], warehouse_lat, warehouse_lon)
    total_distance += distance_to_warehouse
    return_travel = travel_time(distance_to_warehouse, avg_speed)
    return_time = current_time + timedelta(hours=return_travel)
    
    # Check if total work time is within limits
    work_hours = (return_time - truck_route['start_time']).total_seconds() / 3600
    if work_hours > workday_length:
        return False, 0, None
    
    # Calculate extra distance compared to original route
    if len(temp_waypoints) == 0:
        extra_distance = total_distance
    else:
        # Calculate original distance
        orig_distance = 0
        prev_pos = warehouse_coords
        
        for wp in temp_waypoints:
            wp_lat, wp_lon = wp['coords']
            distance = haversine_distance(prev_pos[0], prev_pos[1], wp_lat, wp_lon)
            orig_distance += distance
            prev_pos = (wp_lat, wp_lon)
        
        # Add return to warehouse
        orig_distance += haversine_distance(prev_pos[0], prev_pos[1], warehouse_lat, warehouse_lon)
        
        extra_distance = total_distance - orig_distance
    
    # Create schedule for the new insertion
    new_schedule = {
        'waypoints': new_waypoints,
        'return_time': return_time,
        'total_distance': total_distance
    }
    
    return True, extra_distance, new_schedule

def insert_order_at_position(truck_route, position, order, warehouse_coords, avg_speed, load_unload_hours, extra_distance, new_schedule):
    """Insert an order at a specific position in the route"""
    order_id = order['id']
    order_lat = order['Latitude']
    order_lon = order['Longitude']
    order_quantity = order['quantity']
    
    # Update truck route with new order
    truck_route['orders'].append(order_id)
    truck_route['total_load'] += order_quantity
    truck_route['current_load'] += order_quantity
    truck_route['total_distance'] += extra_distance
    
    # If it's the first order, set up the route
    if len(truck_route['waypoints']) == 0:
        # Distance from warehouse to order
        warehouse_lat, warehouse_lon = warehouse_coords
        distance_to_order = haversine_distance(warehouse_lat, warehouse_lon, order_lat, order_lon)
        
        # Travel time
        travel_hours = travel_time(distance_to_order, avg_speed)
        arrival_time = truck_route['start_time'] + timedelta(hours=travel_hours)
        
        # Adjust arrival time if we're early
        if 'arrival' in new_schedule and new_schedule['arrival'] > arrival_time:
            arrival_time = new_schedule['arrival']
        
        # Service time
        service_end = arrival_time + timedelta(hours=load_unload_hours)
        
        # Create waypoint
        waypoint = {
            'order_id': order_id,
            'coords': (order_lat, order_lon),
            'arrival_time': arrival_time,
            'end_time': service_end,
            'quantity': order_quantity,
            'distance_from_prev': distance_to_order,
            'distance_to_next': 0  # Will be updated later
        }
        
        truck_route['waypoints'].append(waypoint)
        
        # Distance back to warehouse
        distance_to_warehouse = haversine_distance(order_lat, order_lon, warehouse_lat, warehouse_lon)
        truck_route['empty_distance'] = distance_to_warehouse  # Return trip is empty
        
        # Calculate expected return time
        return_hours = travel_time(distance_to_warehouse, avg_speed)
        return_time = service_end + timedelta(hours=return_hours)
        truck_route['end_time'] = return_time
    else:
        # Use the precalculated schedule from check_insertion_feasibility
        truck_route['waypoints'] = new_schedule['waypoints']
        truck_route['end_time'] = new_schedule['return_time']
        
        # Recalculate empty miles (simplified approach - assumes return to warehouse is empty)
        last_wp = truck_route['waypoints'][-1]
        last_lat, last_lon = last_wp['coords']
        warehouse_lat, warehouse_lon = warehouse_coords
        empty_dist = haversine_distance(last_lat, last_lon, warehouse_lat, warehouse_lon)
        truck_route['empty_distance'] = empty_dist

def finalize_route(truck_route, warehouse_coords, avg_speed):
    """Add final return to warehouse if not already added"""
    if not truck_route['waypoints']:
        return
    
    # Make sure end_time is calculated
    if truck_route['end_time'] is None:
        last_waypoint = truck_route['waypoints'][-1]
        last_lat, last_lon = last_waypoint['coords']
        warehouse_lat, warehouse_lon = warehouse_coords
        
        # Calculate return to warehouse
        distance_to_warehouse = haversine_distance(last_lat, last_lon, warehouse_lat, warehouse_lon)
        return_hours = travel_time(distance_to_warehouse, avg_speed)
        return_time = last_waypoint['end_time'] + timedelta(hours=return_hours)
        
        truck_route['end_time'] = return_time
        truck_route['empty_distance'] = distance_to_warehouse
    
    # Add warehouse as final destination (for visualization)
    warehouse_arrival = truck_route['end_time']
    
    warehouse_waypoint = {
        'order_id': 'Warehouse',
        'coords': warehouse_coords,
        'arrival_time': warehouse_arrival,
        'end_time': warehouse_arrival,
        'quantity': 0,
        'distance_from_prev': 0,
        'distance_to_next': 0
    }
    
    # Add only if the last waypoint is not already the warehouse
    last_waypoint = truck_route['waypoints'][-1]
    if last_waypoint['order_id'] != 'Warehouse':
        truck_route['waypoints'].append(warehouse_waypoint)
