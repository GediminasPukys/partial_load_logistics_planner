import streamlit as st
import pandas as pd
from datetime import datetime, time
import os

from data_handler import load_data, validate_data
from optimization import optimize_routes
from visualization import visualize_routes, create_truck_tables, create_summary

def main():
    st.set_page_config(page_title="Truck Fleet Planner", layout="wide")
    st.title('Truck Fleet Daily Work Planner')
    
    # Sidebar for user inputs
    st.sidebar.header('Fleet Parameters')
    num_trucks = st.sidebar.number_input('Number of trucks in fleet', min_value=1, value=5)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        warehouse_lat = st.number_input('Warehouse Latitude', value=55.69)
    with col2:
        warehouse_lon = st.number_input('Warehouse Longitude', value=12.52)
    
    max_capacity = st.sidebar.number_input('Maximum truck capacity', min_value=0.1, value=10.0, help="Maximum load capacity for all trucks")
    avg_speed = st.sidebar.number_input('Average speed (km/h)', min_value=1.0, value=70.0, help="Average truck speed")
    load_unload_time = st.sidebar.number_input('Load/Unload time (minutes)', min_value=1, value=30, help="Time needed for loading/unloading at each location")
    workday_length = st.sidebar.number_input('Workday length (hours)', min_value=1.0, max_value=24.0, value=8.0, help="Maximum working hours per truck")
    
    # Default start time
    default_time = time(8, 0)  # 8:00 AM
    
    # Allow setting individual truck start times
    st.sidebar.header('Truck Start Times')
    use_custom_start_times = st.sidebar.checkbox('Set custom start times for trucks')
    
    truck_start_times = {}
    if use_custom_start_times:
        for i in range(1, min(num_trucks + 1, 11)):  # Limit to avoid UI clutter
            start_time = st.sidebar.time_input(f'Truck {i} start time', default_time)
            truck_start_times[i] = start_time
            
        if num_trucks > 10:
            st.sidebar.info(f"Using default start time ({default_time.strftime('%H:%M')}) for trucks 11-{num_trucks}")
            for i in range(11, num_trucks + 1):
                truck_start_times[i] = default_time
    else:
        # Use the same start time for all trucks
        for i in range(1, num_trucks + 1):
            truck_start_times[i] = default_time
    
    # File upload
    st.header("Upload Orders Data")
    uploaded_file = st.file_uploader("Upload CSV file with orders data", type=['csv'])
    
    use_example_file = st.checkbox("Use example file", value=False)
    if use_example_file:
        uploaded_file = "example_data.csv"
    
    if uploaded_file is not None:
        # Load and validate data
        df = load_data(uploaded_file)
        if df is not None and validate_data(df):
            st.success("Data loaded successfully!")
            
            # Display the data
            st.subheader("Order Data")
            st.dataframe(df.style.format({
                'Available_from': lambda x: x.strftime('%Y-%m-%d %H:%M'),
                'Available_to': lambda x: x.strftime('%Y-%m-%d %H:%M'),
                'Latitude': '{:.6f}',
                'Longitude': '{:.6f}'
            }))
            
            # Run optimization
            optimize_button = st.button('Optimize Routes', type="primary")
            if optimize_button:
                with st.spinner("Optimizing routes..."):
                    warehouse_coords = (warehouse_lat, warehouse_lon)
                    fleet_params = {
                        'num_trucks': num_trucks,
                        'warehouse_coords': warehouse_coords,
                        'max_capacity': max_capacity,
                        'avg_speed': avg_speed,
                        'load_unload_time': load_unload_time,
                        'workday_length': workday_length,
                        'truck_start_times': truck_start_times,
                        'date': datetime.now().date()  # Use current date for context
                    }
                    
                    routes, unplanned_orders, unused_trucks = optimize_routes(df, fleet_params)
                    
                    # Store results in session state
                    st.session_state.routes = routes
                    st.session_state.unplanned_orders = unplanned_orders
                    st.session_state.unused_trucks = unused_trucks
                    st.session_state.warehouse_coords = warehouse_coords
                    st.session_state.df = df
                    st.session_state.optimized = True
                
                st.success("Route optimization completed!")
        else:
            st.error("Please upload a valid CSV file with the required columns.")
    
    # Display results if optimization has been run
    if 'optimized' in st.session_state and st.session_state.optimized:
        st.header('Route Visualization')
        routes = st.session_state.routes
        warehouse_coords = st.session_state.warehouse_coords
        df = st.session_state.df
        
        selected_truck = visualize_routes(routes, warehouse_coords, df)
        
        st.header('Truck Details')
        if selected_truck:
            create_truck_tables(routes, selected_truck, warehouse_coords)
        else:
            st.info("Please select a truck from the dropdown above to view detailed information.")
        
        st.header('Summary')
        create_summary(routes, st.session_state.unplanned_orders, st.session_state.unused_trucks, df)

if __name__ == '__main__':
    main()
