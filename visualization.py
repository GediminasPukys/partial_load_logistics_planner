import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
from datetime import datetime, timedelta
from utils import haversine_distance, format_time, format_duration, generate_color, truck_utilization, calculate_empty_miles

def visualize_routes(routes, warehouse_coords, orders_df):
    """
    Visualize truck routes on a map and allow selection of a specific truck
    
    Args:
        routes: Dictionary of routes for each truck
        warehouse_coords: Coordinates of the warehouse
        orders_df: DataFrame with order information
        
    Returns:
        selected_truck: ID of the selected truck for detailed view
    """
    # Create a dictionary to map order IDs to their details
    order_details = orders_df.set_index('id').to_dict('index')
    
    # Prepare data for the map
    all_points = []
    all_paths = []
    
    # Add warehouse point
    warehouse_point = {
        'position': [warehouse_coords[1], warehouse_coords[0]],  # [lon, lat] for deck.gl
        'icon': 'warehouse',
        'size': 30,
        'color': [255, 0, 0],  # Red for warehouse
        'name': 'Warehouse'
    }
    all_points.append(warehouse_point)
    
    # Available trucks for selection
    active_trucks = [truck_id for truck_id, route in routes.items() if route['orders']]
    
    if not active_trucks:
        st.warning("No routes could be created with the given constraints.")
        return None
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_truck = st.selectbox('Select truck to view details', 
                                     options=active_trucks,
                                     format_func=lambda x: f"Truck {x}")
        
        # Show truck stats
        if selected_truck:
            route = routes[selected_truck]
            st.write(f"**Orders:** {len(route['orders'])}")
            st.write(f"**Total load:** {route['total_load']:.2f}")
            st.write(f"**Total distance:** {route['total_distance']:.2f} km")
            
            # Calculate working hours
            if route['end_time'] and route['start_time']:
                work_hours = (route['end_time'] - route['start_time']).total_seconds() / 3600
                st.write(f"**Working hours:** {format_duration(work_hours)}")
    
    # Generate colors for each truck
    truck_colors = {}
    for i, truck_id in enumerate(active_trucks):
        # Use the utility function to generate colors
        truck_colors[truck_id] = generate_color(i, len(active_trucks), truck_id == selected_truck)
    
    # For each truck, add its route
    for truck_id, route in routes.items():
        if not route['orders']:
            continue
            
        color = truck_colors[truck_id]
        is_selected = (truck_id == selected_truck)
        line_width = 4 if is_selected else 2
        
        # Start from warehouse
        path_points = []
        
        # Add warehouse as starting point
        path_points.append([warehouse_coords[1], warehouse_coords[0]])  # [lon, lat]
        
        # Add each waypoint
        for waypoint in route['waypoints']:
            if waypoint['order_id'] == 'Warehouse':
                continue  # Skip the final return to warehouse in path points
                
            lat, lon = waypoint['coords']
            
            # Add to the path
            path_points.append([lon, lat])
            
            # Add point marker
            point = {
                'position': [lon, lat],
                'icon': 'circle',
                'size': 20 if is_selected else 15,
                'color': color if is_selected else [100, 100, 100],
                'name': f"Order {waypoint['order_id']} ({waypoint['quantity']})",
                'tooltip': f"Order: {waypoint['order_id']}\nQuantity: {waypoint['quantity']}\nArrival: {format_time(waypoint['arrival_time'])}"
            }
            all_points.append(point)
        
        # Add return to warehouse
        path_points.append([warehouse_coords[1], warehouse_coords[0]])
        
        # Create path
        path = {
            'path': path_points,
            'color': color,
            'width': line_width,
            'truck_id': truck_id
        }
        all_paths.append(path)
    
    # Create the deck.gl layers
    icon_layer = pdk.Layer(
        'IconLayer',
        all_points,
        get_position='position',
        get_icon='icon',
        get_size='size',
        get_color='color',
        pickable=True,
        size_scale=1
    )
    
    path_layer = pdk.Layer(
        'PathLayer',
        all_paths,
        get_path='path',
        get_color='color',
        get_width='width',
        pickable=True,
        width_scale=1,
        width_min_pixels=2
    )
    
    # Set the initial view state
    view_state = pdk.ViewState(
        longitude=warehouse_coords[1],
        latitude=warehouse_coords[0],
        zoom=9,
        pitch=0
    )
    
    # Create the deck
    deck = pdk.Deck(
        layers=[path_layer, icon_layer],
        initial_view_state=view_state,
        tooltip={
            'text': '{name}'
        },
        map_style='mapbox://styles/mapbox/light-v10'
    )
    
    # Display the map
    with col2:
        st.pydeck_chart(deck)
    
    return selected_truck

def create_truck_tables(routes, selected_truck, warehouse_coords):
    """
    Create detailed tables for the selected truck
    
    Args:
        routes: Dictionary of routes for each truck
        selected_truck: ID of the selected truck
        warehouse_coords: Coordinates of the warehouse
    """
    if selected_truck is None:
        return
        
    route = routes[selected_truck]
    
    if not route['orders']:
        st.write(f"Truck {selected_truck} has no assigned orders.")
        return
    
    # Create waypoints table
    waypoints_data = []
    
    # Add warehouse start
    warehouse_lat, warehouse_lon = warehouse_coords
    
    # Add each waypoint including the warehouse start
    total_distance = 0
    prev_coords = warehouse_coords
    
    for i, waypoint in enumerate(route['waypoints']):
        lat, lon = waypoint['coords']
        
        # Skip the warehouse waypoint at the end (handled separately)
        if i > 0 and waypoint['order_id'] == 'Warehouse':
            continue
            
        # Calculate distance from previous
        if i == 0:
            distance_from_prev = haversine_distance(warehouse_lat, warehouse_lon, lat, lon)
        else:
            prev_lat, prev_lon = prev_coords
            distance_from_prev = haversine_distance(prev_lat, prev_lon, lat, lon)
        
        total_distance += distance_from_prev
        
        # Format location name
        if waypoint['order_id'] == 'Warehouse':
            location = "Warehouse"
        else:
            location = f"Order {waypoint['order_id']}"
        
        # Add to waypoints data
        wp_data = {
            'Stop': i + 1,
            'Location': location,
            'Arrival': format_time(waypoint['arrival_time']),
            'Departure': format_time(waypoint['end_time']),
            'Coordinates': f"{lat:.6f}, {lon:.6f}",
            'Quantity': waypoint['quantity'],
            'Distance from Previous (km)': round(distance_from_prev, 2),
            'Cumulative Distance (km)': round(total_distance, 2)
        }
        waypoints_data.append(wp_data)
        
        # Update previous coordinates
        prev_coords = (lat, lon)
    
    # Add warehouse return as final stop
    last_lat, last_lon = prev_coords
    distance_to_warehouse = haversine_distance(last_lat, last_lon, warehouse_lat, warehouse_lon)
    total_distance += distance_to_warehouse
    
    warehouse_end = {
        'Stop': len(waypoints_data) + 1,
        'Location': "Warehouse (Return)",
        'Arrival': format_time(route['end_time']),
        'Departure': format_time(route['end_time']),
        'Coordinates': f"{warehouse_lat:.6f}, {warehouse_lon:.6f}",
        'Quantity': 0,
        'Distance from Previous (km)': round(distance_to_warehouse, 2),
        'Cumulative Distance (km)': round(total_distance, 2)
    }
    waypoints_data.append(warehouse_end)
    
    # Create summary
    if route['end_time'] and route['start_time']:
        total_hours = (route['end_time'] - route['start_time']).total_seconds() / 3600
    else:
        total_hours = 0
    
    # Show route information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Truck {selected_truck} Schedule")
        st.write(f"**Start Time:** {format_time(route['start_time'])}")
        st.write(f"**End Time:** {format_time(route['end_time'])}")
        st.write(f"**Total Orders:** {len(route['orders'])}")
    
    with col2:
        st.write(f"**Total Distance:** {route['total_distance']:.2f} km")
        st.write(f"**Total Load:** {route['total_load']:.2f}")
        st.write(f"**Working Hours:** {format_duration(total_hours)}")
    
    # Display waypoints table
    st.subheader("Route Details")
    st.dataframe(pd.DataFrame(waypoints_data), use_container_width=True)
    
    # Create a time-distance chart
    create_route_chart(waypoints_data, route['start_time'], route['end_time'])

def create_route_chart(waypoints_data, start_time, end_time):
    """Create a visual chart of the route timing and distances"""
    if not waypoints_data:
        return
    
    # Prepare data for the chart
    chart_data = []
    
    for wp in waypoints_data:
        # Convert string times to datetime
        arrival_time = datetime.strptime(wp['Arrival'], '%H:%M') if wp['Arrival'] != 'N/A' else None
        departure_time = datetime.strptime(wp['Departure'], '%H:%M') if wp['Departure'] != 'N/A' else None
        
        if arrival_time and departure_time:
            # Add arrival point
            chart_data.append({
                'Stop': wp['Location'],
                'Time': arrival_time.strftime('%H:%M'),
                'Distance': wp['Cumulative Distance (km)'],
                'Type': 'Arrival'
            })
            
            # Add departure point
            chart_data.append({
                'Stop': wp['Location'],
                'Time': departure_time.strftime('%H:%M'),
                'Distance': wp['Cumulative Distance (km)'],
                'Type': 'Departure'
            })
    
    # Create chart if we have data
    if chart_data:
        df = pd.DataFrame(chart_data)
        
        # Create the chart
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X('Time:N', sort=None, title='Time'),
            y=alt.Y('Distance:Q', title='Distance (km)'),
            color=alt.Color('Stop:N', legend=None),
            tooltip=['Stop', 'Time', 'Distance', 'Type']
        ).properties(
            title='Distance-Time Chart',
            width=700,
            height=300
        )
        
        st.altair_chart(chart, use_container_width=True)

def create_summary(routes, unplanned_orders, unused_trucks, orders_df):
    """
    Create a summary of the optimization results
    
    Args:
        routes: Dictionary of routes for each truck
        unplanned_orders: List of orders that couldn't be planned
        unused_trucks: Number of unused trucks
        orders_df: DataFrame with order information
    """
    # Count successful orders
    planned_orders = sum(len(route['orders']) for route in routes.values())
    total_orders = len(orders_df)
    
    # Calculate total distance
    total_distance = sum(route['total_distance'] for route in routes.values())
    
    # Calculate total load
    total_load = sum(route['total_load'] for route in routes.values())
    
    # Calculate average truck utilization
    used_trucks = len(routes) - unused_trucks
    if used_trucks > 0:
        avg_distance = total_distance / used_trucks
        avg_load = total_load / used_trucks
    else:
        avg_distance = 0
        avg_load = 0
    
    # Create summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Planned Orders", f"{planned_orders}/{total_orders}", 
                  f"{planned_orders/total_orders*100:.1f}%" if total_orders > 0 else "0%")
        st.metric("Total Distance", f"{total_distance:.2f} km")
    
    with col2:
        st.metric("Used Trucks", f"{used_trucks}/{len(routes)}", 
                  f"{used_trucks/len(routes)*100:.1f}%" if len(routes) > 0 else "0%")
        st.metric("Avg Distance per Truck", f"{avg_distance:.2f} km")
    
    with col3:
        st.metric("Total Load", f"{total_load:.2f}")
        st.metric("Avg Load per Truck", f"{avg_load:.2f}")
    
    # Create tabs for different summary views
    tab1, tab2, tab3 = st.tabs(["Truck Usage", "Unplanned Orders", "Route Comparison"])
    
    with tab1:
        # Create truck usage summary
        create_truck_usage_summary(routes, total_load)
    
    with tab2:
        # Show unplanned orders if any
        if unplanned_orders:
            st.subheader(f"Unplanned Orders ({len(unplanned_orders)})")
            unplanned_df = orders_df[orders_df['id'].isin(unplanned_orders)]
            st.dataframe(unplanned_df.style.format({
                'Available_from': lambda x: x.strftime('%Y-%m-%d %H:%M'),
                'Available_to': lambda x: x.strftime('%Y-%m-%d %H:%M'),
            }), use_container_width=True)
        else:
            st.success("All orders successfully planned!")
    
    with tab3:
        # Create route comparison chart
        create_route_comparison(routes)

def create_truck_usage_summary(routes, total_load):
    """Create a summary of truck usage"""
    # Gather data for all trucks
    truck_data = []
    
    for truck_id, route in routes.items():
        # Calculate working hours
        if route['end_time'] and route['start_time']:
            work_hours = (route['end_time'] - route['start_time']).total_seconds() / 3600
        else:
            work_hours = 0
        
        # Calculate metrics
        num_orders = len(route['orders'])
        truck_load = route['total_load']
        distance = route['total_distance']
        
        # Add to the dataset
        truck_data.append({
            'Truck': f"Truck {truck_id}",
            'Orders': num_orders,
            'Load': truck_load,
            'Load %': (truck_load / total_load * 100) if total_load > 0 else 0,
            'Distance (km)': distance,
            'Working Hours': work_hours
        })
    
    # Create a DataFrame
    df = pd.DataFrame(truck_data)
    
    # Display the table
    st.dataframe(df.style.format({
        'Load': '{:.2f}',
        'Load %': '{:.1f}%',
        'Distance (km)': '{:.2f}',
        'Working Hours': '{:.2f}'
    }), use_container_width=True)
    
    # Create charts if we have data
    if not df.empty and df['Orders'].sum() > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Load distribution chart
            load_chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('Truck:N', title='Truck'),
                y=alt.Y('Load:Q', title='Load'),
                color=alt.Color('Truck:N', legend=None),
                tooltip=['Truck', 'Load', 'Orders']
            ).properties(
                title='Load Distribution',
                width=350,
                height=300
            )
            
            st.altair_chart(load_chart, use_container_width=True)
        
        with col2:
            # Distance chart
            distance_chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('Truck:N', title='Truck'),
                y=alt.Y('Distance (km):Q', title='Distance (km)'),
                color=alt.Color('Truck:N', legend=None),
                tooltip=['Truck', 'Distance (km)', 'Orders']
            ).properties(
                title='Distance by Truck',
                width=350,
                height=300
            )
            
            st.altair_chart(distance_chart, use_container_width=True)

def create_route_comparison(routes):
    """Create a comparison of routes"""
    # Prepare data for the comparison
    active_routes = {k: v for k, v in routes.items() if v['orders']}
    
    if not active_routes:
        st.info("No active routes to compare.")
        return
    
    # Gather key metrics
    comparison_data = []
    
    for truck_id, route in active_routes.items():
        # Calculate working hours
        if route['end_time'] and route['start_time']:
            work_hours = (route['end_time'] - route['start_time']).total_seconds() / 3600
        else:
            work_hours = 0
        
        # Calculate empty miles percentage
        empty_miles_pct = calculate_empty_miles(route)
        
        comparison_data.append({
            'Truck': f"Truck {truck_id}",
            'Orders': len(route['orders']),
            'Total Distance (km)': route['total_distance'],
            'Empty Miles (%)': empty_miles_pct,
            'Working Hours': work_hours,
            'Avg Dist Per Order (km)': route['total_distance'] / len(route['orders']) if len(route['orders']) > 0 else 0
        })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Display the comparison table
    st.dataframe(df.style.format({
        'Total Distance (km)': '{:.2f}',
        'Empty Miles (%)': '{:.1f}%',
        'Working Hours': '{:.2f}',
        'Avg Dist Per Order (km)': '{:.2f}'
    }), use_container_width=True)
    
    # Create comparison chart
    if not df.empty:
        # Prepare data for a stacked bar chart
        chart_data = []
        
        for _, row in df.iterrows():
            truck = row['Truck']
            total_dist = row['Total Distance (km)']
            empty_pct = row['Empty Miles (%)'] / 100
            
            chart_data.append({
                'Truck': truck,
                'Type': 'Productive Miles',
                'Distance': total_dist * (1 - empty_pct)
            })
            
            chart_data.append({
                'Truck': truck,
                'Type': 'Empty Miles',
                'Distance': total_dist * empty_pct
            })
        
        chart_df = pd.DataFrame(chart_data)
        
        # Create stacked bar chart
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X('Truck:N', title='Truck'),
            y=alt.Y('Distance:Q', title='Distance (km)'),
            color=alt.Color('Type:N', scale=alt.Scale(
                domain=['Productive Miles', 'Empty Miles'],
                range=['#1f77b4', '#ff7f0e']
            )),
            tooltip=['Truck', 'Type', 'Distance']
        ).properties(
            title='Distance Breakdown by Truck',
            width=700,
            height=400
        )
        
        st.altair_chart(chart, use_container_width=True)
