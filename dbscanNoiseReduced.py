import numpy as np
import pandas as pd
import pyodbc
from pymongo import MongoClient
from sklearn.cluster import DBSCAN 
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# READ UNLOADING POINTS AND DESTINATION FROM SOURCE
# ===================================================
# destination_name = "Suzuki Kulana Unloading"
# src_lat, src_lon = 28.431140, 76.655132

# destination_name = "Kherwas"
# src_lat, src_lon = 23.052975, 75.253428
# file_path = "C:/Users/TEST/Downloads/KherwasUnloadingPoints.xlsx"
# df = pd.read_excel(file_path)
# print(df.head())



# MONGO DB CONFIGURATIONS 
# ==========================
client = MongoClient("mongodb://localhost:27017/")  # adjust if needed
db = client["dbscan_db"]
collection = db["destinations"]


# Create Mongo Db Document Structure
destination_doc = {
    "destination_name": destination_name,    #TODO: variables not defined
    "src_lat": src_lat,
    "src_lon": src_lon,
    "ranges": []
}


# SQL CONNECTION
# =================

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=your_server;"
    "DATABASE=your_db;"
    "UID=your_username;"
    "PWD=your_password"
)


# Function to calculate DBSCAN.
def run_dbscan(points, eps_meters, min_samples=20):
    if len(points) == 0:
        return None
    points_rad = np.radians(points)

    db = DBSCAN(
        eps=eps_meters/6371000,   # convert meters to radians
        min_samples=min_samples,
        metric="haversine"
    ).fit(points_rad)
    return db.labels_


# READ COMPANY IDs FROM Resoure File
# ====================================
with open("company_ids.txt") as f:
    company_ids = [line.strip() for line in f if line.strip()]

for company_id in company_ids:
    print(f"Processing company {company_id}...")

    query = f"""
        SELECT 
            company_id,
            destination_name,
            destination_lat,
            destination_lon,
            unloading_lat,
            unloading_lon
        FROM company_locations
        WHERE company_id = ?
    """

    df = pd.read_sql(query, conn, params=[company_id])

    if df.empty:
        print(f"No data found for company {company_id}")
        continue


    # Extract destination info
    destination_name = df["destination_name"].iloc[0]
    src_lat = df["destination_lat"].iloc[0]
    src_lon = df["destination_lon"].iloc[0]

    # Extract unloading points (lat, lon pairs)
    coords = df[["force_lat", "force_long"]].dropna().to_numpy()



    # Transform Unloading points and destination point to Radians
    coords_rad = np.radians(coords)
    src_rad = np.radians([src_lat, src_lon])


    # Calculate Haversine Distance in KM
    dlat = coords_rad[:,0] - src_rad[0]
    dlon = coords_rad[:,1] - src_rad[1]
    a = np.sin(dlat/2.0)**2 + np.cos(src_rad[0]) * np.cos(coords_rad[:,0]) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = 6371 * c 

    print("Distances ->", distances[:5])



    # Define bins(Concentric circle Ideology from destination around destination) with ranges in a Dictionary
    # ( Range bins contains unloading points w.r.t their distance from the destination point )

    # Eg:- All the points within a range of  100m > [Unloading points] <= 500m will be within
    #  the first bin of along with their pre-defined epsilon value(distance of each point w.r.t medoid when creating clusters)
    # These Unloading points are not in radians but simply Lat , long
    range_bins = {
        "0m–100m": (coords[distances <= 0.1], 30),             # eps = 30m
        "100m–500m": (coords[(distances > 0.1) & (distances <= 0.5)], 50), 
        "500m–1000m": (coords[(distances > 0.5) & (distances <= 1.0)], 100),
        "1000–2000m": (coords[(distances > 1.0) & (distances <= 2.0)], 100),
        "2000–3000m": (coords[(distances > 2.0) & (distances <= 3.0)], 100),
        "3000–4000m": (coords[(distances > 3.0) & (distances <= 4.0)], 200),
        "4000–5000m": (coords[(distances > 4.0) & (distances <= 5.0)], 200),
        "5000–6000m": (coords[(distances > 5.0) & (distances <= 6.0)], 200),
        "6000–7000m": (coords[(distances > 6.0) & (distances <= 7.0)], 300),
        "7000–8000m": (coords[(distances > 7.0) & (distances <= 8.0)], 300),
        "8000–9000m": (coords[(distances > 8.0) & (distances <= 9.0)], 300),
    }



    # Assign a colormap list for different rings
    # (This will help in coloring Unloading points w.r.t their position withing range bins)
    colormaps = [
        "tab20", "Set1", "Accent", "Dark2", "Paired",
        "Set2", "tab10", "Pastel1", "Pastel2", "hsv", "terrain"
    ]



    # Create a canvas for laying points over.
    plt.figure(figsize=(10,8))



    # Apply DBSCAN on each bin separately.
    # Example :- (label, (points, eps_m)) will be of type ("100m–500m" , (Array[Unloading points] , 30(epsilon)))
    for i, (label, (points, eps_m)) in enumerate(range_bins.items()):


        # labels will contain cluster ids of each point w.r.t their indices 
        # Points -> [P1 , P2 , P3 , P4] , correspoinding  labels -> [-1 , 0 , 0 ,2]
        labels = run_dbscan(points, eps_meters=eps_m, min_samples=10)


        # range entry is ranges array in mongo document
        range_entry = {
            "range_label": label,
            "clusters": []
        }


        if labels is not None:
            unique, counts = np.unique(labels, return_counts=True)
            cluster_counts = {int(k): int(v) for k, v in zip(unique, counts)}
            print(f"Range {label} -> Cluster sizes: {cluster_counts}")

            colors = ["lightgray"] + list(plt.cm.tab10.colors) * 10    # first color for noise(To remove confusion)
            plt.scatter(                                               # Scatter the clusters over the canvas created earlier.             
            points[:,1], points[:,0],
                c=[colors[l+1] if l >= 0 else colors[0] for l in labels],
                s=30, alpha=0.8, label=f"{label} (eps={eps_m}m)"
            )


                #  This loop is essentially for calculating medoid within a cluster
                # Method of calculation of medoid
                #                 ||
                #                \  /
                # (pairwise distance with all points in cluster , point with sum(pairwise dist) minimum => medoid) 
            for cluster_id in np.unique(labels):



                # np.unique(labels) will give unique label ids eg:- [-1 , 0 , 1 ,2]
                # Cluster points will pick from points array points corresponding to each cluster_id
                cluster_points = points[labels == cluster_id]

                # Do not perform for noise(i.e cluster id = -1)
                if cluster_id == -1:
                    continue

                cluster_points_rad = np.radians(cluster_points)
                dists = pairwise_distances(cluster_points_rad, metric="haversine")
                medoid_idx = np.argmin(dists.sum(axis=0))
                medoid_rad = cluster_points_rad[medoid_idx]   # medoid in radians
                medoid = cluster_points[medoid_idx]    
            


                # This is for drawing a circle around cluster w.r.t th medoid
                # Method of calculation -> distance(haversine) of medoid w.r.t each cluster point
                #  and findig max dist to draw circle around medoid of that radius     
                dlat = cluster_points_rad[:,0] - medoid_rad[0]
                dlon = cluster_points_rad[:,1] - medoid_rad[1]
                a = np.sin(dlat/2.0)**2 + np.cos(medoid_rad[0]) * np.cos(cluster_points_rad[:,0]) * np.sin(dlon/2.0)**2
                c = 2 * np.arcsin(np.sqrt(a))
                radius_km = 6371 * np.max(c) if len(cluster_points) > 1 else 0.05



                # build cluster entry(cluster array in ranges document(Mongo))
                cluster_entry = {
                    "cluster_id": int(cluster_id),
                    "medoid": {
                        "lat": float(medoid[0]),
                        "lon": float(medoid[1])
                    },
                    "radius_km": float(radius_km),
                    "unloading_points": [
                        {"lat": float(p[0]), "lon": float(p[1])} for p in cluster_points
                    ]
                }


                range_entry["clusters"].append(cluster_entry)
                lat_scale = 111.32  # km per degree latitude
                lon_scale = 40075 * np.cos(medoid_rad[0]) / 360.0 # km per degree longitude at this latitude

                radius_deg_lat = radius_km / lat_scale
                radius_deg_lon = radius_km / lon_scale
                radius_deg = max(radius_deg_lat, radius_deg_lon)   # ensure circle fully encloses points


                # radius_deg = radius_km / 111.0
                circle = Circle(
                    (medoid[1], medoid[0]),   # lon, lat
                    radius_deg,
                    edgecolor="green",
                    facecolor="none", lw=1.5, alpha=0.6, ls="--"
                )
                plt.gca().add_patch(circle)

                plt.text(
                medoid[1], medoid[0],
                str(cluster_counts[cluster_id]),
                fontsize=9, fontweight="bold",
                color="black",
                ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.2")
                )
            destination_doc["ranges"].append(range_entry)        
        else:
            print(f"Range {label} -> No points")    


    # --- Insert into Mongo ---
    collection.insert_one(destination_doc)
    print("Destination inserted into MongoDB")        
    plt.scatter(src_lon, src_lat, c="red", marker="*", s=200, label="Source")


    # Add concentric circles (in degrees, approx conversion from km)
    circle_ranges_km = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # same as range bins
    for r in circle_ranges_km:
        circle = Circle(
            (src_lon, src_lat),               # center
            r / 111.0,                        # km → degrees approx (1° ~ 111 km)
            color="black", fill=False, ls="--", lw=0.8, alpha=0.5
        )
        plt.gca().add_patch(circle)


    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"DBSCAN   clustering in distance-based rings")
    plt.legend()
    plt.show()
