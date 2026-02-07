import random
import math
import heapq

# ------------------- PATIENT DATA GENERATION -------------------
def generate_patient_data(num_patients):
    """Generates a list of patient records with ID, severity, and location."""
    patients = []
    for i in range(num_patients):
        patient = {
            "id": i + 1,
            "severity": random.randint(1, 10),  # Random severity level (1-10)
            "location": (random.randint(0, 100), random.randint(0, 100))  # Random coordinates
        }
        patients.append(patient)
    return patients

patients = generate_patient_data(50)  # Generate 50 patients
print("Initial Patient Data in our Database:")
for p in patients:
    print(p)

# ------------------- HOSPITAL DATA GENERATION -------------------
def generate_hospital_data(num_hospitals):
    """Generates hospital data with ID, location, and available beds."""
    hospitals = []
    for i in range(num_hospitals):
        hospital = {
            "id": i + 1,
            "location": (random.randint(0, 100), random.randint(0, 100)),
            "beds": random.randint(0, 5)  # Beds available (can be 0)
        }
        hospitals.append(hospital)
    return hospitals

hospitals = generate_hospital_data(5)  # Generate 5 hospitals
print("\nHospital Data:")
for h in hospitals:
    print(h)

# ------------------- AMBULANCE DATA GENERATION -------------------
ambulances = []
def generate_ambulance_data(num_ambulances=20):
    """Generates a fleet of ambulances with random status and location."""
    for i in range(num_ambulances):
        status = random.choice(['free', 'free', 'busy'])  # More chance to be 'free'
        ambulance = {
            "id": 100 + i,  # Unique ID starting from 100
            "status": status,
            "location": (random.randint(0, 100), random.randint(0, 100))
        }
        ambulances.append(ambulance)
    return ambulances

ambulances = generate_ambulance_data()
for i in range(len(ambulances)):
    print(ambulances[i])

# ------------------- CITY GRAPH (NODES & EDGES) -------------------
# Coordinates for road network nodes
NODE_DATA = """
  1,10,10
  2,30,10
  3,10,30
  4,30,30
  5,50,30
  6,30,50
  7,50,50
  8,70,50
  9,50,5
  10,70,10
  11,90,10
  12,90,30
  13,70,30
  14,90,50
  15,90,70
  16,70,70
  17,50,70
  18,30,70
  19,10,70
  20,10,50
"""

# Road connections (edges) with distances
EDGE_DATA = """
  1,2,20
  1,3,22
  2,4,20
  2,9,25
  3,4,10
  4,5,25
  4,6,30
  5,7,15
  5,9,18
  6,7,10
  7,8,20
  8,10,22
  9,10,30
  10,11,20
  11,12,18
  12,13,12
  13,5,14
  12,14,16
  14,15,18
  15,16,10
  16,17,12
  17,18,14
  18,6,20
  18,19,15
  19,20,10
  20,3,25
"""

# ------------------- SORTING PATIENTS -------------------
def sort(patients):
    """Sorts patients by severity in descending order."""
    return sorted(patients, key=lambda patient: patient['severity'], reverse=True)

# ------------------- SEARCH PATIENT BY SEVERITY -------------------
def search_severity(sorted_patients, target_severity):
    """Finds all patients with a given severity using binary search."""
    found_patients = []
    left, right = 0, len(sorted_patients) - 1
    first_index = -1

    # Find first occurrence of the target severity
    while left <= right:
        mid = (left + right) // 2
        if sorted_patients[mid]['severity'] == target_severity:
            first_index = mid
            right = mid - 1
        elif sorted_patients[mid]['severity'] > target_severity:
            left = mid + 1
        else:
            right = mid - 1

    if first_index == -1:
        return found_patients

    # Find last occurrence of the target severity
    left, right = first_index, len(sorted_patients) - 1
    last_index = -1
    while left <= right:
        mid = (left + right) // 2
        if sorted_patients[mid]['severity'] == target_severity:
            last_index = mid
            left = mid + 1
        elif sorted_patients[mid]['severity'] > target_severity:
            left = mid + 1
        else:
            right = mid - 1

    return sorted_patients[first_index:last_index+1]

# ------------------- DISTANCE CALCULATION -------------------
def calculate_distance(point1, point2):
    """Calculates Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# ------------------- NEAREST HOSPITAL -------------------
def find_nearest_available_hospital(patient, hospitals):
    """Finds the nearest hospital with available beds."""
    closest_hospital = None
    min_distance = float('inf')
    for hospital in hospitals:
        if hospital['beds'] > 0:
            distance = calculate_distance(patient['location'], hospital['location'])
            if distance < min_distance:
                min_distance = distance
                closest_hospital = hospital
    return closest_hospital, min_distance

# ------------------- NEAREST NODE (GRAPH) -------------------
def find_nearest_node(point_coord, node_coords):
    """Finds nearest road network node to a given (x, y) coordinate."""
    nearest_node_id = None
    min_distance = float('inf')
    for node_id, coord in node_coords.items():
        distance = calculate_distance(point_coord, coord)
        if distance < min_distance:
            min_distance = distance
            nearest_node_id = node_id
    return nearest_node_id

# ------------------- NEAREST AMBULANCE -------------------
def find_best_ambulance(patient, ambulance_fleet):
    """Finds nearest available ambulance to the patient."""
    closest_ambulance = None
    min_distance = float('inf')
    for ambulance in ambulance_fleet:
        if ambulance['status'] == 'free':
            distance = calculate_distance(patient['location'], ambulance['location'])
            if distance < min_distance:
                min_distance = distance
                closest_ambulance = ambulance
    return closest_ambulance

# ------------------- DIJKSTRA'S ALGORITHM -------------------
def find_shortest_path(graph, start_node, end_node):
    """Finds shortest path between two nodes using Dijkstra's algorithm."""
    pq = [(0, start_node, [])]  # (distance, current_node, path)
    visited = set()
    while pq:
        dist, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        path = path + [node]
        if node == end_node:
            return path, dist
        for neighbor, weight in graph.get(node, []):
            if neighbor not in visited:
                heapq.heappush(pq, (dist + weight, neighbor, path))
    return [], float('inf')

# ------------------- LOAD GRAPH FROM DATA -------------------
def load_graph_from_data(node_str, edge_str):
    """Parses node and edge data into a graph and node coordinate map."""
    node_coords = {}
    for line in node_str.strip().split('\n'):
        parts = line.split(',')
        node_id, x, y = int(parts[0]), int(parts[1]), int(parts[2])
        node_coords[node_id] = (x, y)

    graph = {node_id: [] for node_id in node_coords}
    for line in edge_str.strip().split('\n'):
        parts = line.split(',')
        u, v, weight = int(parts[0]), int(parts[1]), int(parts[2])
        graph[u].append((v, weight))
        graph[v].append((u, weight))
    return graph, node_coords

# ------------------- MAIN SIMULATION -------------------
city_graph, node_coordinates = load_graph_from_data(NODE_DATA, EDGE_DATA)
print("--- Real-World Ambulance Dispatch Simulation ---")
print("City road network loaded successfully.")

print(50*"*")
sorted_patient = sort(patients)
print("Here is the Sorted Patient Data based on severity:\n")
for p in sorted_patient[:10]:
    print(p)
print(50*"*")

# Ask user for severity search
target = int(input("Enter severity level to search (1-10, 10 = most severe): "))
print(f"\n--- Searching for patients with severity {target} ---")
results = search_severity(sorted_patient, target)

if results:
    print(f"Found {len(results)} patients with severity {target}:")
    for p in results:
        print(p)
else:
    print(f"No patients found with severity {target}.")

print(50*"*")
patient_id = input("Please enter the ID of the high-priority patient: ")
high_priority_patient = next((p for p in sorted_patient if str(p['id']) == patient_id), None)

# Find nearest hospital and ambulance
closest_hospital, distance = find_nearest_available_hospital(high_priority_patient, hospitals)
best_ambulance = find_best_ambulance(high_priority_patient, ambulances)

# Ask if user wants to see available ambulances
ambulance_choice = input("Do you want to see all the available ambulances nearby? (y/n): ").strip().lower()

if ambulance_choice == 'y':
    print("\nAll the available ambulances:")
    for amb in ambulances:
        if amb["status"] == "free":
            print(f"Ambulance {amb['id']} is available at location {amb['location']}")

    # Get coordinates
    ambulance_coordinates = best_ambulance['location']
    patient_coordinates = high_priority_patient['location']
    hospital_coordinates = closest_hospital['location']

    # Map to nearest road network nodes
    ambulance_start_node = find_nearest_node(ambulance_coordinates, node_coordinates)
    patient_node = find_nearest_node(patient_coordinates, node_coordinates)
    hospital_node = find_nearest_node(hospital_coordinates, node_coordinates)

    print(f"\n--- High priority patient ID {high_priority_patient['id']} at location {high_priority_patient['location']} ---")
    print(f"\nClosest Hospital ID: {closest_hospital['id']} at Location {closest_hospital['location']}")
    print(f"Distance from the Patient: {distance:.2f} km")
    print(f"Available Beds in that hospital: {closest_hospital['beds']}")

    print("\nMapping locations to nearest road network intersections...")
    print(f"  - Ambulance Depot -> Node {ambulance_start_node}")
    print(f"  - Patient Pickup  -> Node {patient_node}")
    print(f"  - Hospital        -> Node {hospital_node}")

    print("\nCalculating optimal routes...")
    path_to_patient, dist_to_patient = find_shortest_path(city_graph, ambulance_start_node, patient_node)
    path_to_hospital, dist_to_hospital = find_shortest_path(city_graph, patient_node, hospital_node)

    print(f"\nRoute 1: Ambulance Dispatch Route to Patient")
    print(f"  - Path: {' -> '.join(map(str, path_to_patient))}")
    print(f"  - Distance: {dist_to_patient} km")

    print(f"\nRoute 2: Patient to Hospital")
    print(f"  - Path: {' -> '.join(map(str, path_to_hospital))}")
    print(f"  - Distance: {dist_to_hospital} km")

    print("\n--- Dispatch Summary ---")
    print(f"Total estimated travel distance: {dist_to_patient + dist_to_hospital} km\n")

elif ambulance_choice == 'n':
    print("Thank you for using our service.")
else:
    print("Invalid choice. Please enter 'y' or 'n'.")