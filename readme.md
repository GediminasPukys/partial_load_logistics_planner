# Truck Fleet Daily Work Planner

This Streamlit application helps plan and optimize truck fleet operations for daily deliveries. It allocates orders to trucks while respecting time windows, truck capacities, and other constraints, with the goal of minimizing the number of trucks used and reducing empty miles.

## Features

- **Data Input**: Upload CSV files with order information (ID, time windows, quantity, and coordinates)
- **Fleet Parameters**: Configure the number of trucks, warehouse location, truck capacity, and more
- **Custom Start Times**: Set different start times for each truck to optimize operations
- **Visualization**: Interactive map showing routes and order locations
- **Detailed Analysis**: View truck-specific details and performance metrics
- **Summary Statistics**: Track planned/unplanned orders, truck utilization, and more

## How to Run

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run main.py
   ```

3. Access the application in your web browser (typically at http://localhost:8501)

## Input Data Format

The application expects a CSV file with the following columns:
- `id`: Unique identifier for each order
- `Available_from`: Start of the time window when the order can be serviced (YYYY-MM-DD HH:MM:SS)
- `Available_to`: End of the time window (YYYY-MM-DD HH:MM:SS)
- `quantity`: Order size/volume
- `Latitude`: Delivery location latitude
- `Longitude`: Delivery location longitude

Example:
```
id,Available_from,Available_to,quantity,Latitude,Longitude
1,2025-03-18 07:00:00,2025-03-18 10:00:00,3.25,55.689774,12.519797
2,2025-03-18 07:00:00,2025-03-18 10:00:00,2.5,55.697546,12.585792
```

## Optimization Approach

The application uses a greedy insertion algorithm that:
1. Sorts orders by earliest deadline and distance from warehouse to cluster geographically proximate orders
2. Prioritizes filling existing trucks before using new ones
3. For each order, finds the best truck and insertion position that minimizes additional distance
4. Respects all constraints (capacity, time windows, working hours)
5. Accepts excellent fits immediately to improve overall distance optimization
6. Returns all trucks to the warehouse at the end of their routes

## File Structure

- `main.py`: Main Streamlit application
- `data_handler.py`: Functions for loading and preprocessing data
- `optimization.py`: Route optimization algorithms
- `visualization.py`: Map and table visualization functions
- `utils.py`: Utility functions (distance calculation, time operations, etc.)
- `requirements.txt`: Required Python packages
- `example_data.csv`: Sample data for testing

## Constraints and Assumptions

- Orders must be serviced within their availability windows
- Trucks start and end at the main warehouse
- Trucks can carry partial loads up to their maximum capacity
- Distances are calculated using air path (straight-line distance)
- Each truck has a maximum working day length
- Load/unload operations take a fixed amount of time

## Extending the Application

To improve the optimization algorithm or add new features:

1. For better optimization, consider implementing more sophisticated algorithms:
   - Vehicle Routing Problem (VRP) solvers
   - Genetic algorithms
   - Simulated annealing

2. Additional features could include:
   - Traffic considerations
   - Driver breaks and rest periods
   - Multiple warehouses or depots
   - Order priorities