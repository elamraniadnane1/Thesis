import pandas as pd
import numpy as np
import os
import json
import uuid
import random
import datetime
from pathlib import Path
import toml
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.models import PointStruct
from tqdm import tqdm

# Path definitions
BASE_PATH = Path("C:/Users/Administrator/Desktop/Thesis")
SECRETS_PATH = BASE_PATH / ".streamlit/secrets.toml"
CERCLES_PATH = BASE_PATH / "cercles.csv"
CENTRES_PATH = BASE_PATH / "centres.csv"
ARRONDISSEMENTS_PATH = BASE_PATH / "arrondissements.csv"
COMMUNES_PATH = BASE_PATH / "communes.csv"

# Load OpenAI API key from secrets.toml
secrets = toml.load(SECRETS_PATH)
openai_api_key = secrets.get("openai", {}).get("api_key")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in secrets.toml")

# Set OpenAI API key
openai.api_key = openai_api_key

# Initialize Qdrant client (using local instance)
qdrant_client = QdrantClient(host="localhost", port=6333)

# Helper function to batch-upsert points
def batch_upsert(collection_name, points, batch_size=500):
    """Insert points in batches to avoid payload size limits"""
    total_points = len(points)
    print(f"Upserting {total_points} points in batches of {batch_size}...")
    
    for i in range(0, total_points, batch_size):
        batch = points[i:i+batch_size]
        qdrant_client.upsert(collection_name=collection_name, points=batch)
        print(f"  ✓ Batch {i//batch_size + 1}/{(total_points-1)//batch_size + 1} ({len(batch)} points)")
    
    print(f"✅ Inserted {total_points} points into {collection_name}")

# Generate fake embeddings
def generate_embedding(text, dim=384):
    """Generate a fake embedding vector for text"""
    # Use a deterministic seed based on the text to ensure consistency
    random.seed(hash(text) % 2**32)
    return [random.uniform(-1, 1) for _ in range(dim)]

# Function to convert NaN values to None (for JSON serialization)
def clean_nan(obj):
    if isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(item) for item in obj]
    return obj

# Generate citizen participation metrics using GPT-4o
def generate_participation_metrics(row, location_type):
    """Generate citizen participation metrics for a location using GPT-4o"""
    try:
        # Create a prompt based on the location information
        location_name = row.get('name_fr', '')
        location_name_ar = row.get('name_ar', '')
        region_name = row.get('region_name', '')
        province_name = row.get('province_name', '')
        population = row.get('population', 0)
        
        if pd.isna(population) or population == 0:
            population = random.randint(1000, 10000)
            
        prompt = f"""
        Generate realistic civic participation metrics for {location_name} ({location_name_ar}) in {province_name}, {region_name}, Morocco.
        This is a {location_type} with approximately {population} residents.
        
        Return a JSON object with these fields:
        1. participation_rate: percentage of citizens actively engaged in local governance (float, 1-100)
        2. digital_access: percentage of citizens with internet access (float, 1-100)
        3. yearly_proposals: number of citizen proposals submitted last year (int)
        4. active_projects: number of community projects currently active (int)
        5. completed_projects: number of projects completed in the last 3 years (int)
        6. top_issues: array of 3-5 main concerns for local citizens (strings)
        7. satisfaction_score: citizen satisfaction with local governance (float, 1-10)
        8. participation_channels: array of available participation methods (strings)
        9. last_town_hall: date of the last town hall meeting (YYYY-MM-DD format)
        10. next_election: date of next local election (YYYY-MM-DD format)
        11. youth_engagement: percentage of participants under 30 years old (float, 1-100)
        12. female_participation: percentage of female participants (float, 1-100)
        13. budget_transparency: score for financial transparency (float, 1-10)
        
        Provide only the JSON object with realistic data, no explanations.
        """
        
        # Call OpenAI API to generate metrics
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data generator for Morocco civic participation metrics. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        metrics = json.loads(response.choices[0].message['content'])
        return metrics
    
    except Exception as e:
        print(f"Error generating metrics for {row.get('name_fr', 'unknown')}: {e}")
        # Return fallback metrics if API call fails
        return {
            "participation_rate": random.uniform(5, 40),
            "digital_access": random.uniform(30, 80),
            "yearly_proposals": random.randint(5, 50),
            "active_projects": random.randint(2, 15),
            "completed_projects": random.randint(5, 30),
            "top_issues": ["Infrastructure", "Water access", "Education", "Employment"],
            "satisfaction_score": random.uniform(3, 8),
            "participation_channels": ["Town halls", "Social media", "Local representatives"],
            "last_town_hall": (datetime.datetime.now() - datetime.timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
            "next_election": (datetime.datetime.now() + datetime.timedelta(days=random.randint(30, 730))).strftime("%Y-%m-%d"),
            "youth_engagement": random.uniform(10, 40),
            "female_participation": random.uniform(20, 60),
            "budget_transparency": random.uniform(2, 9)
        }

# Generate additional administrative data
def generate_admin_data(row, location_type):
    """Generate additional administrative data for a location"""
    
    # Get population with proper error handling
    try:
        population_str = row.get('population', '1000')
        if population_str == '' or pd.isna(population_str):
            population = 1000
        else:
            population = float(population_str)
    except (ValueError, TypeError):
        population = 1000
    
    # Generate local officials
    official_count = max(3, int(population / 5000))
    officials = []
    
    for i in range(official_count):
        gender = random.choice(["male", "female"])
        first_name = random.choice([
            "Mohammed", "Ahmed", "Youssef", "Ali", "Omar", "Ibrahim", "Hassan", "Hamza", "Nizar", "Karim",
            "Fatima", "Aisha", "Maryam", "Nora", "Laila", "Samira", "Amina", "Zineb", "Khadija", "Naima"
        ])
        last_name = random.choice([
            "Alaoui", "Benjelloun", "Chaoui", "Bennani", "El Fassi", "Tahiri", "Idrissi", "Berrada", "Tazi", 
            "Chraibi", "Mansouri", "Haddaoui", "Lahlou", "Benmoussa", "Saadaoui", "Ziani", "Khattabi"
        ])
        
        officials.append({
            "id": str(uuid.uuid4()),
            "name": f"{first_name} {last_name}",
            "role": random.choice([
                "Mayor", "Deputy Mayor", "Council Member", "Director of Urban Planning", 
                "Director of Social Affairs", "Chief of Infrastructure", "Financial Officer",
                "Community Relations Officer", "Environmental Officer", "Cultural Affairs Director"
            ]),
            "start_date": (datetime.datetime.now() - datetime.timedelta(days=random.randint(100, 2000))).strftime("%Y-%m-%d"),
            "email": f"{first_name.lower()}.{last_name.lower()}@gov.ma",
            "phone": f"0{random.choice(['6', '7'])}{random.randint(10000000, 99999999)}",
            "education": random.choice(["High School", "Bachelor's", "Master's", "PhD", "Technical Degree"]),
            "expertise": random.choice([
                "Urban Planning", "Finance", "Social Services", "Education", "Healthcare", 
                "Infrastructure", "Environment", "Culture", "Economic Development", "Security"
            ]),
            "local_resident": random.choice([True, True, True, False]),
            "approval_rating": round(random.uniform(30, 95), 1)
        })
    
    # Generate infrastructure data
    try:
        population_str = row.get('population', '1000')
        if population_str == '' or pd.isna(population_str):
            population = 1000
        else:
            population = float(population_str)
    except (ValueError, TypeError):
        population = 1000
    
    infrastructure = {
        "roads_quality": round(random.uniform(2, 9), 1),
        "water_access": round(min(99, max(20, 40 + random.normalvariate(20, 15))), 1),
        "electricity_coverage": round(min(100, max(50, 75 + random.normalvariate(15, 10))), 1),
        "internet_coverage": round(min(100, max(10, 30 + random.normalvariate(30, 20))), 1),
        "public_buildings": max(1, int(population / 2000)),
        "schools": max(1, int(population / 3000)),
        "healthcare_facilities": max(1, int(population / 7000)),
        "parks": max(0, int(population / 10000)),
        "markets": max(1, int(population / 5000)),
        "public_transport": random.choice(["None", "Limited", "Adequate", "Good", "Excellent"]),
        "waste_management": random.choice(["Poor", "Basic", "Standard", "Advanced", "Excellent"]),
        "last_infrastructure_project": (datetime.datetime.now() - datetime.timedelta(days=random.randint(30, 730))).strftime("%Y-%m-%d")
    }
    
    # Generate development indicators
    development = {
        "literacy_rate": round(min(99, max(40, 60 + random.normalvariate(15, 10))), 1),
        "unemployment_rate": round(min(60, max(5, 15 + random.normalvariate(10, 8))), 1),
        "poverty_rate": round(min(70, max(3, 20 + random.normalvariate(10, 15))), 1),
        "household_income": round(random.uniform(2000, 15000), 0),
        "development_index": round(random.uniform(0.2, 0.9), 2),
        "main_economic_activities": random.sample([
            "Agriculture", "Tourism", "Fishing", "Handicrafts", "Services", "Small Industry", 
            "Mining", "Construction", "Commerce", "Public Sector"
        ], k=min(3, max(1, int(population / 5000))))
    }
    
    # Generate budget information
    budget_total = population * random.uniform(500, 2000)
    
    budget = {
        "total_annual": round(budget_total, 0),
        "per_capita": round(budget_total / population if population > 0 else 1000, 0),
        "infrastructure_pct": round(random.uniform(20, 50), 1),
        "social_services_pct": round(random.uniform(10, 30), 1),
        "administration_pct": round(random.uniform(15, 35), 1),
        "education_pct": round(random.uniform(5, 20), 1),
        "health_pct": round(random.uniform(5, 15), 1),
        "other_pct": round(random.uniform(5, 15), 1),
        "fiscal_year": f"{datetime.datetime.now().year}-{datetime.datetime.now().year + 1}",
        "last_audit": (datetime.datetime.now() - datetime.timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d")
    }
    
    # Challenges and opportunities
    challenges = random.sample([
        "Water scarcity", "Unemployment", "Poor infrastructure", "Limited healthcare access",
        "Education quality", "Youth migration", "Aging population", "Environmental degradation",
        "Limited economic opportunities", "Housing shortage", "Transportation", "Digital divide",
        "Climate change impacts", "Social inequality", "Waste management", "Energy access",
        "Agricultural productivity", "Tourism development", "Public safety", "Cultural preservation"
    ], k=random.randint(3, 6))
    
    opportunities = random.sample([
        "Tourism development", "Renewable energy", "Agricultural modernization", "Artisanal exports",
        "Digital economy growth", "Youth entrepreneurship", "Cultural heritage preservation",
        "Infrastructure improvement", "Educational innovation", "Healthcare enhancement",
        "Eco-tourism", "Local product development", "Remote work hub", "Sustainable agriculture",
        "Community-based tourism", "Traditional crafts revival", "Regional cooperation"
    ], k=random.randint(3, 5))
    
    return {
        "officials": officials,
        "infrastructure": infrastructure,
        "development": development,
        "budget": budget,
        "challenges": challenges,
        "opportunities": opportunities
    }

# Generate citizen profiles
def generate_citizen_profiles(row, location_type, count=5):
    """Generate profiles of engaged citizens for a location"""
    pop_val = row.get('population', 1000)
    try:
        if pop_val == '' or pd.isna(pop_val):
            population = 1000
        else:
            population = float(pop_val)
    except (ValueError, TypeError):
        population = 1000

    profiles = []
    
    for i in range(count):
        gender = random.choice(["male", "female"])
        first_name = random.choice([
            "Mohammed", "Ahmed", "Youssef", "Ali", "Omar", "Ibrahim", "Hassan", "Hamza", "Nizar", "Karim",
            "Fatima", "Aisha", "Maryam", "Nora", "Laila", "Samira", "Amina", "Zineb", "Khadija", "Naima"
        ])
        last_name = random.choice([
            "Alaoui", "Benjelloun", "Chaoui", "Bennani", "El Fassi", "Tahiri", "Idrissi", "Berrada", "Tazi", 
            "Chraibi", "Mansouri", "Haddaoui", "Lahlou", "Benmoussa", "Saadaoui", "Ziani", "Khattabi"
        ])
        
        age = random.randint(18, 80)
        
        occupation = random.choice([
            "Teacher", "Farmer", "Shopkeeper", "Student", "Retired", "Civil Servant", "Business Owner",
            "Artisan", "Taxi Driver", "Doctor", "Engineer", "Lawyer", "Homemaker", "Construction Worker",
            "Restaurant Worker", "Fisherman", "Tour Guide", "Security Guard", "Nurse", "Factory Worker"
        ])
        
        engagement_types = random.sample([
            "Town hall participant", "Community leader", "Neighborhood representative", 
            "Local NGO member", "Digital platform user", "Petition organizer", 
            "Volunteer", "Project beneficiary", "Budget monitor", "Youth council member",
            "Women's group member", "Environmental activist", "Cultural preservation advocate",
            "Education committee", "Healthcare advocate", "Small business association", 
            "Agricultural cooperative", "Local journalist", "Social media activist"
        ], k=random.randint(1, 3))
        
        contribution = random.choice([
            "Regular attendance at town meetings", 
            "Organizing community cleanup efforts",
            "Advocating for better schools", 
            "Reporting infrastructure issues",
            "Mobilizing neighbors for local initiatives", 
            "Contributing to local development plans",
            "Monitoring project implementation", 
            "Representing vulnerable community members",
            "Documenting local heritage", 
            "Facilitating community dialogues",
            "Proposing innovative solutions", 
            "Connecting officials with citizens"
        ])
        
        profiles.append({
            "id": str(uuid.uuid4()),
            "name": f"{first_name} {last_name}",
            "age": age,
            "gender": gender,
            "occupation": occupation,
            "years_in_community": min(age - 5, random.randint(1, 50)),
            "education_level": random.choice(["Primary", "Secondary", "University", "None", "Vocational"]),
            "engagement_level": random.choice(["High", "Medium", "Low"]),
            "engagement_types": engagement_types,
            "digital_usage": random.choice(["High", "Medium", "Low", "None"]),
            "contribution": contribution,
            "concerns": random.sample([
                "Water access", "Road quality", "Public safety", "Job opportunities", "Healthcare", 
                "Education", "Housing", "Transportation", "Environmental issues", "Cultural preservation",
                "Market access", "Internet connectivity", "Electricity supply", "Youth engagement",
                "Women's participation", "Local governance", "Agricultural support"
            ], k=random.randint(2, 4)),
            "satisfaction": round(random.uniform(1, 10), 1)
        })
    
    return profiles

# Create Qdrant collections
def create_collections():
    """Create Qdrant collections for Morocco geographic and participation data"""
    collections = {
        "morocco_cercles": 384,
        "morocco_centres": 384,
        "morocco_arrondissements": 384,
        "morocco_communes": 384
    }
    
    for collection_name, vector_size in collections.items():
        if not qdrant_client.collection_exists(collection_name):
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"✅ Created collection: {collection_name}")
        else:
            print(f"ℹ️ Collection already exists: {collection_name}")

# Main function to process CSV files and insert into Qdrant
def main():
    print("🇲🇦 Morocco Geographic Data with Citizen Participation Processor")
    print("="*70)
    
    # Create Qdrant collections
    print("\n📊 Creating Qdrant collections...")
    create_collections()
    
    # Process Cercles
    print("\n🔄 Processing Cercles data...")
    cercles_df = pd.read_csv(CERCLES_PATH, encoding='utf-8')
    cercles_df = cercles_df.fillna("")
    
    cercles_points = []
    for _, row in tqdm(cercles_df.iterrows(), total=len(cercles_df)):
        # Skip rows with missing essential data
        if not row['code'] or not row['name_fr']:
            continue
            
        # Generate embedding
        text_for_embedding = f"{row['name_fr']} {row['name_ar']} {row['region_name']} {row['province_name']}"
        embedding = generate_embedding(text_for_embedding)
        
        # Generate participation metrics using GPT-4o
        participation_metrics = generate_participation_metrics(row, "cercle")
        
        # Generate additional administrative data
        admin_data = generate_admin_data(row, "cercle")
        
        # Generate citizen profiles
        citizen_profiles = generate_citizen_profiles(row, "cercle")
        
        # Prepare payload
        payload = row.to_dict()
        payload = clean_nan(payload)
        
        # Add generated data
        payload["participation_metrics"] = participation_metrics
        payload["administrative_data"] = admin_data
        payload["citizen_profiles"] = citizen_profiles
        payload["location_type"] = "cercle"
        payload["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate a UUID based on the code (for consistent retrieval)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(row['code'])))
        
        cercles_points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
        )
    
    # Insert Cercles data
    batch_upsert("morocco_cercles", cercles_points)
    
    # Process Centres
    print("\n🔄 Processing Centres data...")
    centres_df = pd.read_csv(CENTRES_PATH, encoding='utf-8')
    centres_df = centres_df.fillna("")
    
    centres_points = []
    for _, row in tqdm(centres_df.iterrows(), total=len(centres_df)):
        # Skip rows with missing essential data
        if not row['code'] or not row['name_fr']:
            continue
            
        # Generate embedding
        text_for_embedding = f"{row['name_fr']} {row['name_ar']} {row['region_name']} {row['province_name']} {row['cercle_name']}"
        embedding = generate_embedding(text_for_embedding)
        
        # Generate participation metrics using GPT-4o
        participation_metrics = generate_participation_metrics(row, "centre")
        
        # Generate additional administrative data
        admin_data = generate_admin_data(row, "centre")
        
        # Generate citizen profiles
        citizen_profiles = generate_citizen_profiles(row, "centre")
        
        # Prepare payload
        payload = row.to_dict()
        payload = clean_nan(payload)
        
        # Add generated data
        payload["participation_metrics"] = participation_metrics
        payload["administrative_data"] = admin_data
        payload["citizen_profiles"] = citizen_profiles
        payload["location_type"] = "centre"
        payload["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate a UUID based on the code (for consistent retrieval)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(row['code'])))
        
        centres_points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
        )
    
    # Insert Centres data
    batch_upsert("morocco_centres", centres_points)
    
    # Process Arrondissements
    print("\n🔄 Processing Arrondissements data...")
    arrondissements_df = pd.read_csv(ARRONDISSEMENTS_PATH, encoding='utf-8')
    arrondissements_df = arrondissements_df.fillna("")
    
    arrondissements_points = []
    for _, row in tqdm(arrondissements_df.iterrows(), total=len(arrondissements_df)):
        # Skip rows with missing essential data
        if not row['code'] or not row['name_fr']:
            continue
            
        # Generate embedding
        text_for_embedding = f"{row['name_fr']} {row['name_ar']} {row['region_name']} {row['province_name']}"
        embedding = generate_embedding(text_for_embedding)
        
        # Generate participation metrics using GPT-4o
        participation_metrics = generate_participation_metrics(row, "arrondissement")
        
        # Generate additional administrative data
        admin_data = generate_admin_data(row, "arrondissement")
        
        # Generate citizen profiles - more for urban areas
        citizen_profiles = generate_citizen_profiles(row, "arrondissement", count=8)
        
        # Prepare payload
        payload = row.to_dict()
        payload = clean_nan(payload)
        
        # Add generated data
        payload["participation_metrics"] = participation_metrics
        payload["administrative_data"] = admin_data
        payload["citizen_profiles"] = citizen_profiles
        payload["location_type"] = "arrondissement"
        payload["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate a UUID based on the code (for consistent retrieval)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(row['code'])))
        
        arrondissements_points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
        )
    
    # Insert Arrondissements data
    batch_upsert("morocco_arrondissements", arrondissements_points)
    
    # Process Communes
    print("\n🔄 Processing Communes data...")
    communes_df = pd.read_csv(COMMUNES_PATH, encoding='utf-8')
    communes_df = communes_df.fillna("")
    
    communes_points = []
    for _, row in tqdm(communes_df.iterrows(), total=len(communes_df)):
        # Skip rows with missing essential data
        if not row['code'] or not row['name_fr']:
            continue
            
        # Generate embedding
        text_for_embedding = f"{row['name_fr']} {row['name_ar']} {row['region_name']} {row['province_name']} {row['cercle_name']}"
        embedding = generate_embedding(text_for_embedding)
        
        # Generate participation metrics using GPT-4o
        participation_metrics = generate_participation_metrics(row, "commune")
        
        # Generate additional administrative data
        admin_data = generate_admin_data(row, "commune")
        
        # Generate citizen profiles
        citizen_profiles = generate_citizen_profiles(row, "commune")
        
        # Prepare payload
        payload = row.to_dict()
        payload = clean_nan(payload)
        
        # Add generated data
        payload["participation_metrics"] = participation_metrics
        payload["administrative_data"] = admin_data
        payload["citizen_profiles"] = citizen_profiles
        payload["location_type"] = "commune"
        payload["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate a UUID based on the code (for consistent retrieval)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(row['code'])))
        
        communes_points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
        )
    
    # Insert Communes data
    batch_upsert("morocco_communes", communes_points)
    
    print("\n✅ All data has been processed and inserted into Qdrant collections!")
    print("="*70)
    print("Summary:")
    print(f"- Cercles: {len(cercles_points)} records")
    print(f"- Centres: {len(centres_points)} records")
    print(f"- Arrondissements: {len(arrondissements_points)} records")
    print(f"- Communes: {len(communes_points)} records")
    print("="*70)
    print("The collections are now ready for use in citizen participation applications!")

if __name__ == "__main__":
    main()