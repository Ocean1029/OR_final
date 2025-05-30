import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime, timedelta

def read_interval_data():
    """
    Read all CSV files in the interval_outputs directory and calculate the statistics of each station
    """
    # Get all CSV files
    csv_files = glob.glob('interval_outputs/interval_*.csv')
    
    # Store data for all stations
    all_stations_data = {}
    station_info = {}  # Store the basic information of the station
    
    # Read each file
    for file in csv_files:
        df = pd.read_csv(file)
        # Process each station
        for _, row in df.iterrows():
            station_id = row['sno']
            if station_id not in all_stations_data:
                all_stations_data[station_id] = {
                    'available_rent_bikes': []
                }
                # Store the basic information of the station
                station_info[station_id] = {
                    'sarea': row['sarea'],
                    'sna': row['sna'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'total': row['total']
                }
            all_stations_data[station_id]['available_rent_bikes'].append(row['available_rent_bikes'])
    
    # Calculate the statistics of each station
    station_stats = {}
    for station_id, data in all_stations_data.items():
        station_stats[station_id] = {
            'sarea': station_info[station_id]['sarea'],
            'sna': station_info[station_id]['sna'],
            'latitude': station_info[station_id]['latitude'],
            'longitude': station_info[station_id]['longitude'],
            'total': station_info[station_id]['total'],
            'rent_mean': np.mean(data['available_rent_bikes']),
            'rent_std': np.std(data['available_rent_bikes'])
        }
    
    return station_stats

def generate_instance(station_stats, time_period):
    """
    Generate instance data for a specific time period
    
    Parameters:
    station_stats: station statistics
    time_period: time period (9, 17, or 22)
    """
    # Create data list
    data = []
    
    # Generate data for each station
    for station_id, stats in station_stats.items():
        # Use normal distribution to generate available_rent_bikes
        rent_bikes = np.random.normal(
            loc=stats['rent_mean'],
            scale=stats['rent_std'],
            size=1
        )[0]
        
        # Ensure rent_bikes is non-negative and not exceeding total
        rent_bikes = max(0, min(rent_bikes, stats['total']))
        
        # Calculate return_bikes as total - rent_bikes
        return_bikes = stats['total'] - rent_bikes
        
        # Create row data
        row = {
            'sno': station_id,
            'sarea': stats['sarea'],
            'sna': stats['sna'],
            'latitude': stats['latitude'],
            'longitude': stats['longitude'],
            'total': stats['total'],
            'available_rent_bikes': round(rent_bikes, 1),
            'available_return_bikes': round(return_bikes, 1)
        }
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure columns are in the correct order
    columns = ['sno', 'sarea', 'sna', 'latitude', 'longitude', 'total', 
              'available_rent_bikes', 'available_return_bikes']
    df = df[columns]
    
    return df

def create_directory_structure(base_dir, num_trucks, time_window):
    """
    Create directory structure for a scenario
    """
    scenario_dir = f"{base_dir}/{num_trucks}trucks_{time_window}min"
    time_periods = {
        'morning_9am': 9,
        'evening_5pm': 17,
        'night_10pm': 22
    }
    
    for period_name in time_periods.keys():
        period_dir = f"{scenario_dir}/{period_name}"
        if not os.path.exists(period_dir):
            os.makedirs(period_dir)
    
    return scenario_dir, time_periods

def main():
    # Create base output directory
    base_dir = "generated_instances"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Read actual data and calculate statistics
    print("Reading interval_outputs data...")
    station_stats = read_interval_data()
    print(f"Read data for {len(station_stats)} stations")
    
    # Generate data for 4 scenarios
    scenarios = [
        (2, 30),  # 2 trucks, 30 minutes time window
        (2, 60),  # 2 trucks, 60 minutes time window
        (4, 30),  # 4 trucks, 30 minutes time window
        (4, 60),  # 4 trucks, 60 minutes time window
        (6, 30),  # 6 trucks, 30 minutes time window
        (6, 60)   # 6 trucks, 60 minutes time window
    ]
    
    for num_trucks, time_window in scenarios:
        print(f"\nGenerating data for {num_trucks} trucks, {time_window} minutes time window...")
        
        # Create directory structure
        scenario_dir, time_periods = create_directory_structure(base_dir, num_trucks, time_window)
        
        # Generate 30 instances for each time period
        for period_name, hour in time_periods.items():
            print(f"Generating instances for {period_name}...")
            for i in range(30):
                # Generate data
                df = generate_instance(
                    station_stats=station_stats,
                    time_period=hour
                )
                
                # Save to CSV file
                filename = f"{scenario_dir}/{period_name}/instance_{i+1}.csv"
                df.to_csv(filename, index=False)
            print(f"Generated 30 instances for {period_name}")

if __name__ == "__main__":
    main() 