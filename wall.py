import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, Set
import matplotlib.patches as mpatches

@dataclass
class LaserMeasurement:
   """
   Represents a single laser measurement
   
   Attributes:
       distance: Measured distance to obstacle
       true_distance: Actual distance to wall
       state: Measurement state ('wall', 'hole', 'edge', 'oor')
       confidence: Measurement confidence [0,1]
   """
   distance: float
   true_distance: float  
   state: str
   confidence: float

class DSBeliefMasses:
   """
   Implements Dempster-Shafer belief mass operations
   """
   def __init__(self):
       # Initialize empty belief masses for all focal elements
       self.masses = {
           frozenset(['wall']): 0.0,
           frozenset(['hole']): 0.0,
           frozenset(['wall', 'edge']): 0.0,
           frozenset(['hole', 'edge']): 0.0,
           frozenset(['wall', 'hole']): 0.0,
           frozenset(['wall', 'hole', 'edge']): 0.0  # Full set (Θ)
       }
   
   def update_mass(self, focal_element: Set[str], mass: float):
       """
       Update mass for a given focal element
       
       Args:
           focal_element: Set of hypotheses
           mass: New mass value [0,1]
       """
       self.masses[frozenset(focal_element)] = mass

   def combine_dempster(self, other: 'DSBeliefMasses', 
                    discount_factor: float = 0.95,
                    memory_factor: float = 0.8) -> 'DSBeliefMasses':
        """
        Combine beliefs with memory effect
        
        Args:
            memory_factor: Weight of historical beliefs [0,1]
        """
        result = DSBeliefMasses()
        theta = frozenset(['wall', 'hole', 'edge'])
        
        # Combine with memory effect
        for A in self.masses:
            old_mass = self.masses[A] * memory_factor
            new_mass = other.masses.get(A, 0) * (1 - memory_factor)
            result.masses[A] = (old_mass + new_mass) * discount_factor
            
        # Normalize
        total_mass = sum(result.masses.values())
        if total_mass > 0:
            for A in result.masses:
                result.masses[A] /= total_mass
                
        return result

def simulate_laser_measurement(true_distance: float, 
                            wall_pattern_period: float,
                            noise_std: float = 0.01) -> LaserMeasurement:
   """
   Simulate laser measurement hitting a wall with periodic holes
   
   Args:
       true_distance: Actual distance to wall
       wall_pattern_period: Period of wall-hole pattern
       noise_std: Standard deviation of measurement noise
       
   Returns:
       LaserMeasurement instance with simulated reading
   """
   # Use random position on wall for each measurement
   relative_pos = np.random.uniform(0, wall_pattern_period)
   
   # Define regions (30% holes, 10% edges on each side)
   hole_width = wall_pattern_period * 0.3
   edge_width = wall_pattern_period * 0.1
   
   # Determine measurement type and value based on position
   if relative_pos < edge_width:
       state = 'edge'
       # Edge measurements are biased towards holes
       distance = true_distance + np.random.normal(0.05, noise_std)
       confidence = 0.6
   elif relative_pos < (edge_width + hole_width):
       state = 'hole'
       distance = float('inf')  # Out of range
       confidence = 0.9
   elif relative_pos < (wall_pattern_period - edge_width):
       state = 'wall'
       distance = true_distance + np.random.normal(0, noise_std)
       confidence = 0.9
   else:
       state = 'edge'
       # Edge measurements are biased towards wall
       distance = true_distance + np.random.normal(-0.05, noise_std)
       confidence = 0.6
       
   return LaserMeasurement(distance, true_distance, state, confidence)

def measurement_to_belief(measurement: LaserMeasurement) -> DSBeliefMasses:
   """
   Convert laser measurement to belief masses
   
   Args:
       measurement: LaserMeasurement instance
       
   Returns:
       DSBeliefMasses instance representing beliefs from measurement
   """
   belief = DSBeliefMasses()
   
   if measurement.distance == float('inf'):
       # Strong belief in hole for out-of-range measurements
       belief.update_mass({'hole'}, 0.8)
       belief.update_mass({'hole', 'edge'}, 0.1)
       belief.update_mass({'wall', 'hole', 'edge'}, 0.1)
   else:
       error = abs(measurement.distance - measurement.true_distance)
       if error < 0.01:
           # Strong belief in wall for accurate measurements
           belief.update_mass({'wall'}, 0.8)
           belief.update_mass({'wall', 'edge'}, 0.1)
           belief.update_mass({'wall', 'hole', 'edge'}, 0.1)
       else:
           # Probably edge for measurements with error
           belief.update_mass({'edge'}, 0.6)
           belief.update_mass({'wall', 'edge'}, 0.2)
           belief.update_mass({'hole', 'edge'}, 0.1)
           belief.update_mass({'wall', 'hole', 'edge'}, 0.1)
   
   return belief

def plot_belief_evolution(beliefs_history, measurements_history):
   """
   Plot evolution of belief masses and measurements over time
   
   Args:
       beliefs_history: List of DSBeliefMasses instances over time
       measurements_history: List of LaserMeasurement instances over time
   """
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
   
   # Plot belief masses evolution
   times = list(range(len(beliefs_history)))
   masses_history = {
       'wall': [], 'hole': [], 'edge': [],
       'wall,edge': [], 'hole,edge': [], 'all': []
   }
   
   # Extract mass values for each hypothesis
   for belief in beliefs_history:
       masses_history['wall'].append(belief.masses.get(frozenset(['wall']), 0))
       masses_history['hole'].append(belief.masses.get(frozenset(['hole']), 0))
       masses_history['edge'].append(belief.masses.get(frozenset(['edge']), 0))
       masses_history['wall,edge'].append(
           belief.masses.get(frozenset(['wall', 'edge']), 0))
       masses_history['hole,edge'].append(
           belief.masses.get(frozenset(['hole', 'edge']), 0))
       masses_history['all'].append(
           belief.masses.get(frozenset(['wall', 'hole', 'edge']), 0))
   
   # Define colors for different hypotheses
   colors = {
       'wall': 'red', 'hole': 'blue', 'edge': 'green',
       'wall,edge': 'orange', 'hole,edge': 'purple', 'all': 'gray'
   }
   
   # Plot each hypothesis
   for key in masses_history:
       ax1.plot(times, masses_history[key], label=key, color=colors[key])
   
   ax1.set_xlabel('Measurement number')
   ax1.set_ylabel('Belief mass')
   ax1.set_title('Evolution of belief masses')
   ax1.grid(True)
   ax1.legend()
   
   # Plot measurements
   distances = [m.distance if m.distance != float('inf') else 3.0 
               for m in measurements_history]
   colors_map = {'wall': 'red', 'hole': 'blue', 'edge': 'green'}
   scatter_colors = [colors_map[m.state] for m in measurements_history]
   
   ax2.scatter(times, distances, c=scatter_colors, alpha=0.6)
   ax2.set_xlabel('Measurement number')
   ax2.set_ylabel('Distance (m)')
   ax2.set_title('Laser measurements')
   ax2.grid(True)
   
   # Add legend for measurements
   legend_elements = [mpatches.Patch(color=colors_map[key], label=key)
                     for key in colors_map]
   ax2.legend(handles=legend_elements)
   
   plt.tight_layout()
   plt.show()

def main():
   """Main simulation loop"""
   # Simulation parameters
   start_distance = 2.0  # Initial distance (meters)
   end_distance = 1.0    # Final distance (meters)
   wall_pattern_period = 0.2  # Wall pattern period (meters)
   n_measurements = 500  # Number of measurements
   
   # Calculate distance change per step
   distance_step = (start_distance - end_distance) / (n_measurements - 1)
   
   # Initialize tracking variables
   accumulated_belief = None
   beliefs_history = []
   measurements_history = []
   
   print(f"Simulating movement towards grated wall from {start_distance}m to {end_distance}m")
   print("\nFirst 5 measurements:")
   
   for i in range(n_measurements):
       # Calculate current distance
       current_distance = start_distance - (i * distance_step)
       
       # Get measurement
       measurement = simulate_laser_measurement(current_distance, wall_pattern_period)
       measurements_history.append(measurement)
       
       # Convert to belief masses
       belief = measurement_to_belief(measurement)
       
       # Update accumulated belief
       if accumulated_belief is None:
           accumulated_belief = belief
       else:
           accumulated_belief = accumulated_belief.combine_dempster(belief)
       
       beliefs_history.append(accumulated_belief)
       
       # Print first 5 measurements
       if i < 5:
           print(f"\nMeasurement {i+1}:")
           print(f"Distance to wall: {current_distance:.3f}m")
           print(f"State: {measurement.state}")
           print(f"Measured distance: {measurement.distance}m")
           print(f"Measurement confidence: {measurement.confidence:.2f}")
           print("Belief masses:")
           for focal_element, mass in belief.masses.items():
               if mass > 0:
                   print(f"  {set(focal_element)}: {mass:.3f}")
   
   print("\nFinal belief masses after", n_measurements, "measurements:")
   for focal_element, mass in accumulated_belief.masses.items():
       if mass > 0:
           print(f"  {set(focal_element)}: {mass:.3f}")
   
   # Visualize results
   plot_belief_evolution(beliefs_history, measurements_history)

if __name__ == "__main__":
   main()