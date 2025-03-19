import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
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
    optimization_objective = fleet_params.get('optimization_objective', 'distance')
    optimize_start_times = fleet_params.get('optimize_start_times', False)

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

    # Add a flag for overnight time windows
    orders['is_overnight'] = orders['Available_to'].dt.date > orders['Available_from'].dt.date

    # Create a dictionary for fast lookup of order time windows
    order_windows = {row['id']: (row['Available_from'], row['Available_to'])
                     for _, row in orders.iterrows()}

    # Filter out orders with invalid time windows
    valid_orders = orders[orders['Available_from'] < orders['Available_to']]
    if len(valid_orders) < len(orders):
        st.warning(f"Excluded {len(orders) - len(valid_orders)} orders with invalid time windows.")

    orders = valid_orders

    # Add an informational message about overnight orders
    overnight_orders = orders[orders['is_overnight']]
    if not overnight_orders.empty:
        st.info(f"Processing {len(overnight_orders)} orders with overnight time windows.")

    # If optimizing start times, analyze order time windows first
    if optimize_start_times:
        truck_start_times = calculate_optimal_start_times(orders, truck_start_times,
                                                          warehouse_coords, avg_speed,
                                                          load_unload_hours, workday_length,
                                                          current_date)

        # Update routes with optimized start times
        for i in range(1, num_trucks + 1):
            if i in truck_start_times:
                start_time = truck_start_times[i]
                start_datetime = combine_date_time(current_date, start_time)
                routes[i]['start_time'] = start_datetime

    # Sort orders based on optimization objective
    if optimization_objective == 'distance':
        # For distance minimization, we want to cluster orders geographically
        # Sort by time window first (still important), then by distance from warehouse

        # Add distance from warehouse column for sorting
        warehouse_lat, warehouse_lon = warehouse_coords
        orders['distance_from_warehouse'] = orders.apply(
            lambda row: haversine_distance(warehouse_lat, warehouse_lon, row['Latitude'], row['Longitude']),
            axis=1
        )

        # Sort by time window end, then by distance from warehouse (to cluster nearby orders)
        orders = orders.sort_values(['Available_to', 'distance_from_warehouse'], ascending=[True, True])
    else:
        # Default sorting: by earliest availability window end time (urgent orders first)
        # Then by quantity (larger orders first to ensure they are planned)
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

        # Special handling for distance optimization objective
        if optimization_objective == 'distance':
            # Try to insert order into trucks that are already in use first
            # before starting a new truck - this helps minimize total vehicle distance
            active_trucks = [truck_id for truck_id, route in routes.items() if route['orders']]
            inactive_trucks = [truck_id for truck_id, route in routes.items() if not route['orders']]

            # Sort trucks by current load to try to fill trucks more completely
            active_trucks.sort(key=lambda truck_id: routes[truck_id]['current_load'] / max_capacity, reverse=True)

            # Create a sorted list with active trucks first, then inactive ones
            sorted_trucks = active_trucks + inactive_trucks
        else:
            # Default sorting
            sorted_trucks = list(range(1, num_trucks + 1))

        # Try trucks in the sorted order
        for truck_id in sorted_trucks:
            truck_route = routes[truck_id]

            # Skip if truck is already full
            if truck_route['current_load'] + order_quantity > max_capacity:
                continue

            # Try inserting at different positions in the route
            positions_to_try = range(len(truck_route['waypoints']) + 1)

            for position in positions_to_try:
                # Check feasibility and calculate extra distance for this insertion
                feasible, extra_dist, new_schedule = check_insertion_feasibility(
                    truck_route, position, order, warehouse_coords,
                    avg_speed, load_unload_hours, workday_length, order_windows
                )

                if feasible and extra_dist < min_extra_distance:
                    min_extra_distance = extra_dist
                    best_truck = truck_id
                    best_insertion_position = position
                    best_schedule = new_schedule

                    # For distance optimization, if this is an excellent fit (very small extra distance),
                    # accept it immediately without checking all other options
                    if optimization_objective == 'distance' and extra_dist < 5.0:  # Under 5km extra distance
                        break

            # If we found an excellent fit, stop checking other trucks
            if optimization_objective == 'distance' and best_truck == truck_id and min_extra_distance < 5.0:
                break

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

    # Second pass: Try to assign any remaining unplanned orders to unused trucks
    if unplanned_orders and any(not route['orders'] for route in routes.values()):
        st.info("Running second optimization pass for unplanned orders...")

        # Get unused trucks
        unused_truck_ids = [truck_id for truck_id, route in routes.items() if not route['orders']]

        # Get unplanned orders
        unplanned_orders_data = orders[orders['id'].isin(unplanned_orders)]

        # Try to assign each unplanned order to an unused truck
        for _, order in unplanned_orders_data.iterrows():
            order_id = order['id']

            # Find best unused truck
            best_truck = None
            min_distance = float('inf')

            for truck_id in unused_truck_ids:
                truck_route = routes[truck_id]

                # Skip if truck is no longer unused (assigned in this second pass)
                if truck_route['orders']:
                    continue

                # Check if this order can be assigned to this truck
                feasible, distance, schedule = check_insertion_feasibility(
                    truck_route, 0, order, warehouse_coords,
                    avg_speed, load_unload_hours, workday_length, order_windows
                )

                if feasible and distance < min_distance:
                    best_truck = truck_id
                    min_distance = distance
                    best_schedule = schedule

            # Assign to best truck if found
            if best_truck is not None:
                truck_route = routes[best_truck]

                # Insert the order
                insert_order_at_position(
                    truck_route, 0, order,
                    warehouse_coords, avg_speed, load_unload_hours,
                    min_distance, best_schedule
                )

                # Mark as planned
                if order_id in unplanned_orders:
                    unplanned_orders.remove(order_id)

                # Remove this truck from unused trucks list if it now has orders
                if truck_route['orders'] and best_truck in unused_truck_ids:
                    unused_truck_ids.remove(best_truck)

    # Finalize routes: add return to warehouse if not already added
    for truck_id, truck_route in routes.items():
        if truck_route['orders']:
            finalize_route(truck_route, warehouse_coords, avg_speed)

    # Count unused trucks
    unused_trucks = sum(1 for truck_route in routes.values() if not truck_route['orders'])

    return routes, unplanned_orders, unused_trucks


def calculate_optimal_start_times(orders, default_start_times, warehouse_coords,
                                  avg_speed, load_unload_hours, workday_length, current_date):
    """
    Calculate optimal start times for trucks based on order time windows,
    ensuring complete coverage of early morning and late night time windows
    while minimizing travel distance.

    Args:
        orders: DataFrame of orders
        default_start_times: Dictionary of default start times by truck ID
        warehouse_coords: Coordinates of the warehouse
        avg_speed: Average truck speed in km/h
        load_unload_hours: Time for loading/unloading in hours
        workday_length: Maximum work day length in hours
        current_date: The planning date

    Returns:
        Dictionary of optimized start times by truck ID
    """
    from datetime import datetime, timedelta, time
    import numpy as np
    import pandas as pd
    import streamlit as st
    from utils import haversine_distance, travel_time

    # Create a list of existing truck IDs
    truck_ids = list(default_start_times.keys())
    optimized_start_times = {}

    # If no orders, return default start times
    if len(orders) == 0:
        return default_start_times

    # Calculate travel time from warehouse to each order
    warehouse_lat, warehouse_lon = warehouse_coords

    # Add travel time information to orders
    orders['travel_time_from_warehouse'] = orders.apply(
        lambda row: travel_time(
            haversine_distance(warehouse_lat, warehouse_lon, row['Latitude'], row['Longitude']),
            avg_speed
        ),
        axis=1
    )

    # Add distance from warehouse for clustering and proximity analysis
    orders['distance_from_warehouse'] = orders.apply(
        lambda row: haversine_distance(warehouse_lat, warehouse_lon, row['Latitude'], row['Longitude']),
        axis=1
    )

    # Create a working copy for time calculations
    orders_for_analysis = orders.copy()

    # Add a column to convert datetime to a float representing hours since midnight
    def datetime_to_hours(dt):
        return dt.hour + dt.minute / 60 + dt.second / 3600

    # Extract hours from datetime for easier calculations
    orders_for_analysis['from_hour'] = orders['Available_from'].apply(datetime_to_hours)
    orders_for_analysis['to_hour'] = orders['Available_to'].apply(datetime_to_hours)

    # Handle overnight windows
    overnight_mask = orders['Available_to'].dt.date > orders['Available_from'].dt.date

    # For overnight windows, add 24 hours to the to_hour
    orders_for_analysis.loc[overnight_mask, 'to_hour'] += 24

    # Create forced time brackets to ensure complete 24-hour coverage
    time_brackets = [
        {'name': 'Very Early Morning', 'start': 0, 'end': 5, 'min_trucks': 1},
        {'name': 'Early Morning', 'start': 5, 'end': 8, 'min_trucks': 1},
        {'name': 'Morning', 'start': 8, 'end': 11, 'min_trucks': 0},
        {'name': 'Midday', 'start': 11, 'end': 14, 'min_trucks': 0},
        {'name': 'Afternoon', 'start': 14, 'end': 17, 'min_trucks': 0},
        {'name': 'Evening', 'start': 17, 'end': 20, 'min_trucks': 1},
        {'name': 'Night', 'start': 20, 'end': 24, 'min_trucks': 1}
    ]

    # Identify orders that fall primarily or exclusively in each time bracket
    for bracket in time_brackets:
        # Find orders where a significant portion of their time window falls in this bracket
        bracket_orders = orders_for_analysis[
            # Orders that start in this bracket
            ((orders_for_analysis['from_hour'] >= bracket['start']) &
             (orders_for_analysis['from_hour'] < bracket['end'])) |
            # Orders that end in this bracket
            ((orders_for_analysis['to_hour'] > bracket['start']) &
             (orders_for_analysis['to_hour'] <= bracket['end'])) |
            # Orders that completely contain this bracket
            ((orders_for_analysis['from_hour'] <= bracket['start']) &
             (orders_for_analysis['to_hour'] >= bracket['end']))
            ]

        # Find orders exclusive to this bracket (or largely confined to it)
        exclusive_orders = bracket_orders[
            # Window is less than 6 hours (not spanning multiple brackets)
            (bracket_orders['to_hour'] - bracket_orders['from_hour'] < 6) &
            # At least half the window is in this bracket
            (((bracket_orders['to_hour'].clip(upper=bracket['end']) -
               bracket_orders['from_hour'].clip(lower=bracket['start'])) /
              (bracket_orders['to_hour'] - bracket_orders['from_hour'])) >= 0.5)
            ]

        # Store the counts in the bracket dict
        bracket['total_orders'] = len(bracket_orders)
        bracket['exclusive_orders'] = len(exclusive_orders)
        bracket['orders_list'] = bracket_orders
        bracket['exclusive_list'] = exclusive_orders

    # Display time bracket analysis
    st.text("Analyzing order distribution across time brackets...")

    for bracket in time_brackets:
        st.text(f"{bracket['name']} ({bracket['start']:02d}:00-{bracket['end']:02d}:00): " +
                f"{bracket['total_orders']} orders, {bracket['exclusive_orders']} exclusive")

    # Determine how many trucks to allocate to each bracket
    total_trucks = len(truck_ids)
    unallocated_trucks = total_trucks

    # First, allocate minimum required trucks to each bracket
    for bracket in time_brackets:
        bracket['allocated_trucks'] = bracket['min_trucks']
        unallocated_trucks -= bracket['min_trucks']

    # Then allocate remaining trucks proportionally based on exclusive order counts
    if unallocated_trucks > 0:
        total_exclusive_orders = sum(b['exclusive_orders'] for b in time_brackets)

        if total_exclusive_orders > 0:
            for bracket in time_brackets:
                # Calculate proportional allocation
                proportion = bracket['exclusive_orders'] / total_exclusive_orders
                additional_trucks = max(0, round(proportion * unallocated_trucks))
                bracket['allocated_trucks'] += additional_trucks

            # Check if we've allocated all trucks
            allocated_total = sum(b['allocated_trucks'] for b in time_brackets)

            # If we have any leftover, allocate to brackets with the most total orders
            if allocated_total < total_trucks:
                remaining = total_trucks - allocated_total
                sorted_brackets = sorted(time_brackets, key=lambda x: x['total_orders'], reverse=True)

                for i in range(remaining):
                    sorted_brackets[i % len(sorted_brackets)]['allocated_trucks'] += 1
        else:
            # If no exclusive orders, distribute evenly
            for i, bracket in enumerate(time_brackets):
                if i < unallocated_trucks:
                    bracket['allocated_trucks'] += 1

    # Optimize start times within each bracket
    assigned_truck_count = 0

    for bracket in time_brackets:
        if bracket['allocated_trucks'] == 0:
            continue

        # Get orders for this bracket
        bracket_orders = bracket['orders_list']

        if len(bracket_orders) == 0:
            # Skip if no orders in this bracket
            continue

        # For brackets with few orders, check geographic clustering
        if len(bracket_orders) <= bracket['allocated_trucks'] * 3:
            # Sort by distance to create geographically proximate assignments
            bracket_orders = bracket_orders.sort_values('distance_from_warehouse')

            # Divide orders into clusters based on allocated trucks
            clusters = []
            cluster_size = max(1, len(bracket_orders) // bracket['allocated_trucks'])

            for i in range(bracket['allocated_trucks']):
                start_idx = i * cluster_size
                end_idx = (i + 1) * cluster_size if i < bracket['allocated_trucks'] - 1 else len(bracket_orders)
                if start_idx < len(bracket_orders):
                    clusters.append(bracket_orders.iloc[start_idx:end_idx])
                else:
                    # Empty cluster if we've run out of orders
                    clusters.append(pd.DataFrame())

            # For each cluster, find the best start time
            for cluster_idx, cluster in enumerate(clusters):
                if len(cluster) == 0:
                    # Use the middle of the bracket for empty clusters
                    best_hour = (bracket['start'] + bracket['end']) / 2
                else:
                    # Find the best start time for this cluster
                    best_hour = find_best_start_time(
                        cluster, bracket['start'], bracket['end'],
                        avg_speed, load_unload_hours, workday_length, 0.5
                    )

                # Convert to time object
                hour = int(best_hour)
                minute = int((best_hour - hour) * 60)

                # Assign to next available truck ID
                if assigned_truck_count < total_trucks:
                    truck_id = truck_ids[assigned_truck_count]
                    optimized_start_times[truck_id] = time(hour, minute)
                    assigned_truck_count += 1
        else:
            # For brackets with many orders, use more sophisticated clustering

            # Try to identify clusters with similar geographic locations
            from sklearn.cluster import KMeans

            # Prepare coordinates for clustering
            coords = bracket_orders[['Latitude', 'Longitude']].values

            # Determine number of clusters (use allocated trucks or fewer if not many orders)
            n_clusters = min(bracket['allocated_trucks'], max(1, len(bracket_orders) // 5))

            if len(coords) >= n_clusters:
                # Use K-means to create geographic clusters
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                bracket_orders['cluster'] = kmeans.fit_predict(coords)

                # Process each geographic cluster
                for i in range(n_clusters):
                    cluster = bracket_orders[bracket_orders['cluster'] == i]

                    if len(cluster) > 0:
                        # Find the best start time for this cluster
                        best_hour = find_best_start_time(
                            cluster, bracket['start'], bracket['end'],
                            avg_speed, load_unload_hours, workday_length, 0.5
                        )

                        # Convert to time object
                        hour = int(best_hour)
                        minute = int((best_hour - hour) * 60)

                        # Assign to next available truck ID
                        if assigned_truck_count < total_trucks:
                            truck_id = truck_ids[assigned_truck_count]
                            optimized_start_times[truck_id] = time(hour, minute)
                            assigned_truck_count += 1
            else:
                # Not enough orders for clustering, use simpler approach
                best_hour = (bracket['start'] + bracket['end']) / 2

                # Assign trucks allocated to this bracket
                for i in range(bracket['allocated_trucks']):
                    if assigned_truck_count < total_trucks:
                        truck_id = truck_ids[assigned_truck_count]

                        # Stagger start times slightly within the bracket
                        hour_offset = i * 0.5  # 30 minute intervals
                        adjusted_hour = min(bracket['end'] - 0.5, best_hour + hour_offset)

                        hour = int(adjusted_hour)
                        minute = int((adjusted_hour - hour) * 60)

                        optimized_start_times[truck_id] = time(hour, minute)
                        assigned_truck_count += 1

    # Ensure all trucks have been assigned
    if assigned_truck_count < total_trucks:
        # Distribute remaining trucks across time brackets that need more coverage
        sorted_brackets = sorted(time_brackets,
                                 key=lambda x: x['total_orders'] / (
                                     x['allocated_trucks'] if x['allocated_trucks'] > 0 else 1),
                                 reverse=True)

        for bracket in sorted_brackets:
            if assigned_truck_count >= total_trucks:
                break

            # Assign an additional truck to this high-demand bracket
            best_hour = (bracket['start'] + bracket['end']) / 2
            hour = int(best_hour)
            minute = int((best_hour - hour) * 60)

            truck_id = truck_ids[assigned_truck_count]
            optimized_start_times[truck_id] = time(hour, minute)
            assigned_truck_count += 1

    # Ensure every truck has a start time
    for truck_id in truck_ids:
        if truck_id not in optimized_start_times:
            optimized_start_times[truck_id] = default_start_times[truck_id]

    # Log the final truck allocations
    bracket_allocations = {b['name']: 0 for b in time_brackets}
    for truck_id, start_time in optimized_start_times.items():
        hour = start_time.hour + start_time.minute / 60
        for bracket in time_brackets:
            if bracket['start'] <= hour < bracket['end']:
                bracket_allocations[bracket['name']] += 1
                break

    # Display the final allocation
    st.info("Optimized truck start time distribution:")
    for bracket_name, count in bracket_allocations.items():
        st.text(f"  {bracket_name}: {count} trucks")

    return optimized_start_times


def find_best_start_time(orders_df, min_hour, max_hour, avg_speed, load_unload_hours, workday_length, step=0.5):
    """
    Find the best start time for a given set of orders within a specified time range.

    Args:
        orders_df: DataFrame containing orders
        min_hour: Minimum hour to consider
        max_hour: Maximum hour to consider
        avg_speed: Average speed in km/h
        load_unload_hours: Time for loading/unloading in hours
        workday_length: Maximum work day length in hours
        step: Time increment to test (in hours)

    Returns:
        best_hour: The best starting hour (float)
    """
    # Test different start times within the range
    test_hours = np.arange(min_hour, max_hour, step)

    best_hour = (min_hour + max_hour) / 2  # Default to middle of range
    best_score = -1

    for hour in test_hours:
        # Evaluate how many orders can be reached with this start time
        reachable_orders = 0
        total_distance = 0

        for _, order in orders_df.iterrows():
            # Calculate when we would arrive at this order
            arrival_hour = hour + order['travel_time_from_warehouse']

            # Check if we arrive within the time window
            if arrival_hour <= order['to_hour']:
                # If we arrive too early, we can wait until the window opens
                effective_arrival = max(arrival_hour, order['from_hour'])

                # Check if we can finish service and return within workday length
                service_end = effective_arrival + load_unload_hours
                return_travel = order['travel_time_from_warehouse']
                total_time = service_end + return_travel - hour

                if total_time <= workday_length:
                    reachable_orders += 1
                    total_distance += 2 * order['distance_from_warehouse']  # Out and back

        # Calculate score - prioritize coverage but consider distance
        if reachable_orders > 0:
            # Score formula emphasizes order coverage but includes distance efficiency
            coverage_score = reachable_orders / len(orders_df) * 100
            distance_score = 100 - (total_distance / (reachable_orders * 100))
            score = (coverage_score * 0.8) + (distance_score * 0.2)

            if score > best_score:
                best_score = score
                best_hour = hour

    return best_hour

def check_insertion_feasibility(truck_route, position, order, warehouse_coords, avg_speed, load_unload_hours,
                                workday_length, order_windows=None):
    """
    Check if inserting an order at a specific position in the route is feasible
    and calculate the extra distance needed

    Args:
        truck_route: The current route
        position: Position to insert the order
        order: The order to insert
        warehouse_coords: Coordinates of the warehouse
        avg_speed: Average speed in km/h
        load_unload_hours: Time for loading/unloading in hours
        workday_length: Maximum work day length in hours
        order_windows: Dictionary mapping order IDs to their time windows

    Returns:
        feasible: Whether the insertion is feasible
        extra_distance: Additional distance required
        new_schedule: New schedule details if feasible
    """
    order_id = order['id']
    order_lat = order['Latitude']
    order_lon = order['Longitude']
    order_quantity = order['quantity']
    order_from = order['Available_from']
    order_to = order['Available_to']

    # Create dictionary of time windows if not provided
    if order_windows is None:
        order_windows = {order_id: (order_from, order_to)}
    elif order_id not in order_windows:
        order_windows[order_id] = (order_from, order_to)

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
        # Using direct datetime comparison which handles overnight windows correctly
        if arrival_time > order_to:
            # If we arrive after the time window ends, this insertion is not feasible
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

    # Add time window information to existing waypoints before inserting new one
    for wp in temp_waypoints:
        if 'from_time' not in wp and 'to_time' not in wp:
            wp_order_id = wp['order_id']
            if wp_order_id == 'Warehouse':
                wp['from_time'] = None
                wp['to_time'] = None
            elif wp_order_id in order_windows:
                wp['from_time'], wp['to_time'] = order_windows[wp_order_id]

    # If inserting in an existing route, need to recalculate the entire schedule
    new_waypoints = temp_waypoints.copy()

    # Create the new waypoint
    new_waypoint = {
        'order_id': order['id'],
        'coords': (order_lat, order_lon),
        'arrival_time': None,  # Will calculate
        'end_time': None,  # Will calculate
        'quantity': order_quantity,
        'distance_from_prev': 0,  # Will calculate
        'distance_to_next': 0,  # Will calculate
        'from_time': order_from,  # Store time window
        'to_time': order_to  # Store time window
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

        # Get order ID and time window for this waypoint
        wp_order_id = wp['order_id']
        wp_from_time = wp.get('from_time')
        wp_to_time = wp.get('to_time')

        # If time window not stored in waypoint, try to get it from order_windows
        if (wp_from_time is None or wp_to_time is None) and wp_order_id in order_windows:
            wp_from_time, wp_to_time = order_windows[wp_order_id]
            # Store in waypoint for future use
            wp['from_time'] = wp_from_time
            wp['to_time'] = wp_to_time

        # Check time window constraints (skip warehouse)
        if wp_order_id != 'Warehouse' and wp_from_time is not None and wp_to_time is not None:
            # This comparison works for overnight windows because datetime objects handle date transitions correctly
            if arrival_time > wp_to_time:
                return False, 0, None

            # If we arrive before the time window starts, wait
            if arrival_time < wp_from_time:
                arrival_time = wp_from_time

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


def insert_order_at_position(truck_route, position, order, warehouse_coords, avg_speed, load_unload_hours,
                             extra_distance, new_schedule):
    """Insert an order at a specific position in the route"""
    order_id = order['id']
    order_lat = order['Latitude']
    order_lon = order['Longitude']
    order_quantity = order['quantity']
    order_from = order['Available_from']
    order_to = order['Available_to']

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
            'distance_to_next': 0,  # Will be updated later
            'from_time': order_from,  # Store time window
            'to_time': order_to  # Store time window
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
        'distance_to_next': 0,
        'from_time': None,  # No time window for warehouse
        'to_time': None  # No time window for warehouse
    }

    # Add only if the last waypoint is not already the warehouse
    last_waypoint = truck_route['waypoints'][-1]
    if last_waypoint['order_id'] != 'Warehouse':
        truck_route['waypoints'].append(warehouse_waypoint)