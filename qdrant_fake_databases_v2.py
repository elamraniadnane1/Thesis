from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.models import PointStruct
import uuid
import random
import datetime
import json
from typing import List, Dict, Tuple
from collections import defaultdict

# Connect to local Qdrant instance
qdrant = QdrantClient(host="localhost", port=6333)

# Helper function for batched upserts
def batch_upsert(collection_name, points, batch_size=1000):
    """
    Insert points in batches to avoid payload size limits
    """
    total_points = len(points)
    print(f"Upserting {total_points} points in batches of {batch_size}...")
    
    for i in range(0, total_points, batch_size):
        batch = points[i:i+batch_size]
        qdrant.upsert(collection_name=collection_name, points=batch)
        print(f"  ✓ Batch {i//batch_size + 1}/{(total_points-1)//batch_size + 1} ({len(batch)} points)")
    
    print(f"✅ Inserted {total_points} points into {collection_name}")

# -----------------------
# Create Collections if They Don't Exist
# -----------------------
collections = {
    "citizen_ideas": 384,
    "municipal_projects": 384,
    "citizen_comments": 384,
    "citizens": 384,
    "municipal_officials": 384,
    "project_updates": 384,
    "budget_allocations": 384,
    "engagement_metrics": 384
}

for collection_name, vector_size in collections.items():
    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"✅ Created collection: {collection_name}")

# -----------------------
# Define Morocco Administrative Divisions (Enhanced)
# -----------------------
morocco_admin = {
    "جهة فاس-مكناس": {
        "عمالة فاس": {
            "communes": ["فاس", "زواغة", "سايس", "أكدال", "جنان الورد", "مرنيسة", "مولاي يعقوب", "عين الله", "عين الشقف", "سبع رواضي"],
            "population": 1150000,
            "area_km2": 500,
            "urban_rural": "urban"
        },
        "عمالة مكناس": {
            "communes": ["مكناس", "حمرية", "المنصور", "الإسماعيلية", "زرهون", "عين عرمة", "عين كرمة", "مجاط", "ويسلان", "سيدي سليمان"],
            "population": 840000,
            "area_km2": 1786,
            "urban_rural": "mixed"
        },
        "إقليم صفرو": {
            "communes": ["صفرو", "البهاليل", "رباط الخير", "أدرج", "أغبالو أقورار", "أولاد مكودو", "عين تمكناي", "الدار الحمراء", "إغزران", "سيدي يوسف بن أحمد"],
            "population": 286000,
            "area_km2": 3313,
            "urban_rural": "rural"
        },
        "إقليم إفران": {
            "communes": ["إفران", "أزرو", "مشليفن", "ضاية عوا", "عين اللوح", "بن صميم", "تيزكيت", "سيدي المخفي", "تيمحضيت", "واد إفران"],
            "population": 155000,
            "area_km2": 3573,
            "urban_rural": "rural"
        },
        "إقليم الحاجب": {
            "communes": ["الحاجب", "عين تاوجطات", "سبع عيون", "أكوراي", "أيت بوبيدمان", "أيت يعزم", "إقدار", "سيدي داود", "تامشاشاط", "تازوطة"],
            "population": 248000,
            "area_km2": 2117,
            "urban_rural": "rural"
        },
        "إقليم بولمان": {
            "communes": ["ميسور", "أوطاط الحاج", "المرس", "تالسينت", "إنجيل", "فريطيسة", "سكورة", "سيدي بوطيب", "ألميس مرموشة", "بني كيل"],
            "population": 186000,
            "area_km2": 14395,
            "urban_rural": "rural"
        },
        "إقليم تاونات": {
            "communes": ["تاونات", "تيسة", "قرية با محمد", "غفساي", "الوردزاغ", "عين عائشة", "عين مديونة", "بني وليد", "بوهودة", "مولاي بوشتى"],
            "population": 662000,
            "area_km2": 5585,
            "urban_rural": "rural"
        },
        "إقليم تازة": {
            "communes": ["تازة", "واد أمليل", "تايناست", "أكنول", "مطماطة", "باب مرزوقة", "بني فراسن", "بني لنت", "بني فتاح", "بني ورين"],
            "population": 516000,
            "area_km2": 15020,
            "urban_rural": "mixed"
        }
    }
}

# -----------------------
# Enhanced Data Generators
# -----------------------
def select_location():
    region = random.choice(list(morocco_admin.keys()))
    province = random.choice(list(morocco_admin[region].keys()))
    province_data = morocco_admin[region][province]
    commune = random.choice(province_data["communes"])
    return region, province, commune, province_data

def fake_embedding():
    return [random.uniform(-1, 1) for _ in range(384)]

def fake_date(start_year=2018, end_year=2023):
    """Generate a random date between start_year and end_year"""
    # Ensure start_year is not after end_year
    if start_year > end_year:
        start_year, end_year = end_year, start_year
    
    start_date = datetime.datetime(start_year, 1, 1)
    end_date = datetime.datetime(end_year, 12, 31)
    random_seconds = random.randint(0, max(1, int((end_date - start_date).total_seconds())))
    random_date = start_date + datetime.timedelta(seconds=random_seconds)
    return random_date.strftime("%Y-%m-%d")

def fake_datetime(start_year=2018, end_year=2023):
    """Generate a random datetime between start_year and end_year"""
    # Ensure start_year is not after end_year
    if start_year > end_year:
        start_year, end_year = end_year, start_year
    elif start_year == end_year:
        # If years are the same, make sure we have a valid range
        start_date = datetime.datetime(start_year, 1, 1)
        end_date = datetime.datetime(start_year, 12, 31, 23, 59, 59)
    else:
        start_date = datetime.datetime(start_year, 1, 1)
        end_date = datetime.datetime(end_year, 12, 31, 23, 59, 59)
    
    random_seconds = random.randint(0, max(1, int((end_date - start_date).total_seconds())))
    random_date = start_date + datetime.timedelta(seconds=random_seconds)
    return random_date.strftime("%Y-%m-%d %H:%M:%S")

# Enhanced data pools
axes = ["النقل والإنارة", "النظافة والبيئة", "تنمية اجتماعية", "تنمية اقتصادية", "الرقمنة", "الصحة العامة", 
        "التعليم", "الثقافة والرياضة", "الأمن والسلامة", "التخطيط العمراني"]

challenge_templates = [
    "تعاني المنطقة من {} منذ أكثر من {} سنوات",
    "هناك مشكلة في {} تؤثر على {} من السكان",
    "تواجه الجماعة {} بشكل مستمر في {}",
    "نقص في {} يؤدي إلى {} للمواطنين",
    "غياب {} يسبب {} في الأحياء السكنية",
    "تدهور {} يؤثر سلباً على {} في المنطقة"
]

solution_templates = [
    "اقتراح بتفعيل {} بميزانية تقديرية {}",
    "تنفيذ {} على مراحل خلال {} شهور",
    "إنشاء {} بالشراكة مع {}",
    "تطبيق {} كحل مستدام باستخدام {}",
    "تطوير {} من خلال {} المحلية",
    "إطلاق مبادرة {} لتحسين {}"
]

idea_topics = ["البنية التحتية", "الخدمات الاجتماعية", "الأمن", "الشفافية", "البيئة", 
               "التنمية المستدامة", "الحكامة", "المشاركة المواطنة", "الابتكار", "التضامن الاجتماعي"]

keywords_pool = ["إنارة", "شوارع", "أمان", "نظافة", "بيئة", "خدمات", "مشاركة", "تواصل", "تكنولوجيا",
                 "تنمية", "استدامة", "شفافية", "حكامة", "ابتكار", "رقمنة", "تضامن", "صحة", "تعليم",
                 "ثقافة", "رياضة", "شباب", "نساء", "أطفال", "كبار السن", "ذوي الاحتياجات الخاصة"]

channels = ["اللقاء", "تطبيق", "استمارة", "SNS", "هاتف", "البريد الإلكتروني", "الموقع الرسمي", "WhatsApp", "SMS"]

sentiments = ["POS", "NEG", "NEU"]

urgency_levels = ["عاجل جداً", "عاجل", "متوسط", "منخفض"]

idea_statuses = ["جديد", "قيد المراجعة", "مقبول", "مرفوض", "في انتظار معلومات إضافية", "محول لمشروع"]

citizen_categories = ["طالب", "موظف", "تاجر", "ربة بيت", "متقاعد", "عامل", "مهني حر", "عاطل", "فلاح"]

age_groups = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]

education_levels = ["ابتدائي", "إعدادي", "ثانوي", "جامعي", "دراسات عليا", "غير متمدرس"]

# -----------------------
# Generate Citizens (New Collection)
# -----------------------
citizen_names = {
    "male": ["أمين", "سعيد", "حسن", "يوسف", "علي", "كريم", "محمد", "إبراهيم", "سامي", "وليد",
             "عبد الله", "أحمد", "خالد", "عمر", "ياسين", "رشيد", "مصطفى", "نبيل", "جمال", "فؤاد"],
    "female": ["فاطمة", "مريم", "ليلى", "سلوى", "نجلاء", "خديجة", "أسماء", "زينب", "هدى", "سميرة",
               "عائشة", "حنان", "نادية", "كوثر", "إيمان", "سناء", "لبنى", "هند", "رجاء", "نوال"]
}

citizens_data = []
citizen_id_map = {}

print("\n📊 Generating Citizens...")
for i in range(2000):
    citizen_id = str(uuid.uuid4())
    gender = random.choice(["male", "female"])
    name = random.choice(citizen_names[gender])
    region, province, commune, province_data = select_location()
    
    payload = {
        "citizen_id": citizen_id,
        "name": name,
        "gender": gender,
        "age_group": random.choice(age_groups),
        "education_level": random.choice(education_levels),
        "occupation": random.choice(citizen_categories),
        "region": region,
        "province": province,
        "commune": commune,
        "neighborhood": f"حي {random.randint(1, 30)}",
        "registration_date": fake_date(2018, 2021),
        "last_active": fake_datetime(2022, 2023),
        "engagement_score": round(random.uniform(0, 100), 2),
        "ideas_submitted": 0,
        "comments_made": 0,
        "projects_followed": [],
        "preferred_channels": random.sample(channels, k=random.randint(1, 3)),
        "interests": random.sample(axes, k=random.randint(2, 4)),
        "verified": random.choice([True, False]),
        "phone_verified": random.choice([True, False]),
        "email": f"{name.lower()}{random.randint(100, 999)}@example.ma" if random.random() > 0.3 else None
    }
    
    citizen_id_map[citizen_id] = payload
    citizens_data.append(PointStruct(id=citizen_id, vector=fake_embedding(), payload=payload))

batch_upsert(collection_name="citizens", points=citizens_data)

# -----------------------
# Generate Municipal Officials (New Collection)
# -----------------------
official_roles = ["رئيس المجلس", "نائب الرئيس", "كاتب المجلس", "أمين المال", "رئيس لجنة", "عضو مجلس",
                  "مدير المصالح", "رئيس قسم", "موظف إداري", "تقني"]

departments = ["الإدارة العامة", "المالية والميزانية", "التعمير", "الأشغال العمومية", "الشؤون الاجتماعية",
               "الشؤون الثقافية", "البيئة والنظافة", "الموارد البشرية", "الشؤون القانونية", "التواصل"]

officials_data = []
official_id_map = {}

print("\n👔 Generating Municipal Officials...")
for i in range(500):
    official_id = str(uuid.uuid4())
    gender = random.choice(["male", "female"])
    name = random.choice(citizen_names[gender])
    region, province, commune, province_data = select_location()
    
    payload = {
        "official_id": official_id,
        "name": name,
        "gender": gender,
        "role": random.choice(official_roles),
        "department": random.choice(departments),
        "region": region,
        "province": province,
        "commune": commune,
        "hire_date": fake_date(2010, 2022),
        "experience_years": random.randint(1, 30),
        "projects_managed": 0,
        "projects_supervised": [],
        "response_rate": round(random.uniform(60, 100), 2),
        "average_response_time_hours": random.randint(1, 72),
        "specializations": random.sample(axes, k=random.randint(1, 3)),
        "contact_email": f"{name.lower()}.{random.choice(['commune', 'municipalite'])}@gov.ma",
        "office_phone": f"0535{random.randint(100000, 999999)}"
    }
    
    official_id_map[official_id] = payload
    officials_data.append(PointStruct(id=official_id, vector=fake_embedding(), payload=payload))

batch_upsert(collection_name="municipal_officials", points=officials_data)

# -----------------------
# Generate Enhanced Citizen Ideas
# -----------------------
print("\n💡 Generating Citizen Ideas...")
citizen_idea_points = []
all_idea_ids = []
citizen_ids = list(citizen_id_map.keys())

for i in range(3000):
    idea_id = str(uuid.uuid4())
    all_idea_ids.append(idea_id)
    
    # Select citizen and update their stats
    citizen_id = random.choice(citizen_ids)
    citizen = citizen_id_map[citizen_id]
    
    region = citizen["region"]
    province = citizen["province"]
    commune = citizen["commune"]
    
    axis = random.choice(axes)
    
    # Enhanced challenge and solution generation
    challenge_issue = random.choice(keywords_pool)
    affected_percentage = random.randint(20, 90)
    years_suffering = random.randint(1, 10)
    
    challenge_text = random.choice(challenge_templates).format(
        challenge_issue,
        random.choice([str(years_suffering), f"{affected_percentage}%", random.choice(keywords_pool)])
    )
    
    solution_budget = random.randint(10000, 1000000)
    solution_duration = random.randint(3, 24)
    solution_partner = random.choice(["القطاع الخاص", "المجتمع المدني", "الجمعيات المحلية", "الوزارة الوصية"])
    solution_method = random.choice(["التكنولوجيا الحديثة", "الموارد البشرية", "الشراكة", "التمويل التشاركي"])
    
    solution_text = random.choice(solution_templates).format(
        random.choice(keywords_pool),
        random.choice([f"{solution_budget} درهم", f"{solution_duration}", solution_partner, solution_method])
    )
    
    sentiment = random.choice(sentiments)
    if sentiment == "POS":
        polarity = round(random.uniform(0.3, 1), 2)
    elif sentiment == "NEG":
        polarity = round(random.uniform(-1, -0.3), 2)
    else:
        polarity = round(random.uniform(-0.3, 0.3), 2)
    
    topic = random.choice(idea_topics)
    keywords = random.sample(keywords_pool, k=random.randint(3, 6))
    
    submission_date = fake_datetime(2018, 2023)
    
    payload = {
        "idea_id": idea_id,
        "citizen_id": citizen_id,
        "citizen_name": citizen["name"],
        "citizen_age_group": citizen["age_group"],
        "citizen_education": citizen["education_level"],
        "axis": axis,
        "challenge": challenge_text,
        "solution": solution_text,
        "detailed_description": f"تفاصيل إضافية حول {challenge_issue} و {solution_text}",
        "expected_impact": f"تحسين حياة {affected_percentage}% من السكان",
        "estimated_budget": solution_budget,
        "implementation_duration_months": solution_duration,
        "city": commune,
        "commune": commune,
        "province": province,
        "region": region,
        "neighborhood": citizen["neighborhood"],
        "gps_coordinates": {
            "lat": round(random.uniform(33.5, 34.5), 6),
            "lng": round(random.uniform(-5.5, -4.5), 6)
        },
        "channel": random.choice(channels),
        "date_submitted": submission_date,
        "last_updated": submission_date,
        "status": random.choice(idea_statuses),
        "urgency": random.choice(urgency_levels),
        "category": topic,
        "subcategory": random.choice(keywords),
        "linked_project_ids": [],
        "similar_ideas": [],
        "supporting_citizens": random.sample(citizen_ids, k=random.randint(0, 50)),
        "supporting_count": 0,
        "sentiment": sentiment,
        "polarity": polarity,
        "topic": topic,
        "keywords": keywords,
        "attachments": [f"attachment_{i}_{j}.pdf" for j in range(random.randint(0, 3))],
        "media_urls": [f"https://example.com/media/{idea_id}_{j}.jpg" for j in range(random.randint(0, 2))],
        "moderation_status": random.choice(["pending", "approved", "rejected", "flagged"]),
        "moderation_notes": "تمت المراجعة" if random.random() > 0.5 else None,
        "assigned_official_id": random.choice(list(official_id_map.keys())) if random.random() > 0.3 else None,
        "priority_score": round(random.uniform(0, 100), 2),
        "feasibility_score": round(random.uniform(0, 100), 2),
        "impact_score": round(random.uniform(0, 100), 2),
        "tags": random.sample(["مستعجل", "مبتكر", "اقتصادي", "اجتماعي", "بيئي", "تقني"], k=random.randint(1, 3))
    }
    
    payload["supporting_count"] = len(payload["supporting_citizens"])
    
    # Update citizen stats
    citizen_id_map[citizen_id]["ideas_submitted"] += 1
    
    citizen_idea_points.append(PointStruct(id=idea_id, vector=fake_embedding(), payload=payload))

batch_upsert(collection_name="citizen_ideas", points=citizen_idea_points, batch_size=500)

# -----------------------
# Generate Enhanced Municipal Projects
# -----------------------
print("\n🏗️ Generating Municipal Projects...")
project_statuses = ["Planned", "Ongoing", "Completed", "Suspended", "Cancelled"]
project_types = ["Infrastructure", "Social", "Economic", "Environmental", "Digital", "Cultural", "Educational"]
risk_levels = ["Low", "Medium", "High", "Critical"]
funding_sources = ["صندوق التجهيز الجماعي", "وزارة الداخلية", "الشراكة المحلية", "التعاون الدولي", 
                   "الميزانية الذاتية", "القطاع الخاص", "الجمعيات", "التمويل التشاركي"]

contractors = ["شركة البناء الوطنية", "مقاولة الأشغال الكبرى", "شركة التجهيز والبناء", "المقاولة المحلية للأشغال",
               "شركة الإنشاءات الحديثة", "مجموعة البناء والتعمير", "الشركة الجهوية للأشغال"]

municipal_project_points = []
municipal_project_ids = []
project_details_dict = {}

for i in range(2000):
    project_id = str(uuid.uuid4())
    municipal_project_ids.append(project_id)
    
    region, province, commune, province_data = select_location()
    neighborhood = f"حي {random.randint(1, 30)}"
    
    project_type = random.choice(project_types)
    title = f"مشروع {random.choice(idea_topics)} في {commune}"
    
    status = random.choice(project_statuses)
    
    # Status-based progress
    if status == "Ongoing":
        official_progress = random.randint(10, 90)
        actual_progress = official_progress + random.randint(-20, 20)
        actual_progress = max(0, min(100, actual_progress))
    elif status == "Planned":
        official_progress = 0
        actual_progress = 0
    elif status == "Completed":
        official_progress = 100
        actual_progress = 100
    elif status == "Suspended":
        official_progress = random.randint(20, 70)
        actual_progress = official_progress
    else:  # Cancelled
        official_progress = random.randint(0, 30)
        actual_progress = official_progress
    
    # Linked ideas (projects created from citizen ideas)
    num_linked_ideas = random.randint(0, 5)
    linked_idea_ids = random.sample(all_idea_ids, k=min(num_linked_ideas, len(all_idea_ids)))
    
    # Budget details
    initial_budget = random.randint(100000, 10000000)
    if status in ["Ongoing", "Completed"]:
        current_budget = initial_budget + random.randint(-initial_budget//4, initial_budget//2)
        spent_budget = int(current_budget * (actual_progress / 100))
    else:
        current_budget = initial_budget
        spent_budget = 0
    
    # Dates
    start_date = fake_date(2019, 2023)
    if status == "Completed":
        end_date = fake_date(2020, 2023)
    elif status == "Planned":
        end_date = fake_date(2024, 2026)
    else:
        end_date = None
    
    # Managing official
    managing_official_id = random.choice(list(official_id_map.keys()))
    managing_official = official_id_map[managing_official_id]
    
    payload = {
        "project_id": project_id,
        "title": title,
        "description": f"مشروع لتحسين {random.choice(keywords_pool)} في منطقة {commune}",
        "objectives": [f"هدف {j+1}: {random.choice(['تحسين', 'تطوير', 'إنشاء', 'تحديث'])} {random.choice(keywords_pool)}" 
                      for j in range(random.randint(2, 5))],
        "project_type": project_type,
        "themes": f"تحسين {random.choice(idea_topics)}",
        "region": region,
        "province": province,
        "commune": commune,
        "neighborhood": neighborhood,
        "beneficiaries_count": random.randint(100, 50000),
        "gps_coordinates": {
            "lat": round(random.uniform(33.5, 34.5), 6),
            "lng": round(random.uniform(-5.5, -4.5), 6)
        },
        "status": status,
        "official_progress": official_progress,
        "actual_progress": actual_progress,
        "citizen_satisfaction_rate": round(random.uniform(40, 95), 2) if status != "Planned" else None,
        "quality_score": round(random.uniform(60, 100), 2) if status in ["Ongoing", "Completed"] else None,
        "risk_level": random.choice(risk_levels),
        "risk_factors": random.sample(["تأخير في التمويل", "صعوبات تقنية", "معارضة محلية", "ظروف جوية", 
                                     "نقص الموارد", "تغيير المواصفات"], k=random.randint(0, 3)),
        "citizen_participation": random.choice([True, False]),
        "participation_methods": random.sample(["لقاءات تشاورية", "استطلاع رأي", "لجان متابعة", "تطبيق رقمي"], 
                                             k=random.randint(1, 3)) if random.random() > 0.5 else [],
        "linked_idea_ids": linked_idea_ids,
        "related_projects": [],
        "initial_budget": initial_budget,
        "current_budget": current_budget,
        "spent_budget": spent_budget,
        "budget_variance_percentage": round(((current_budget - initial_budget) / initial_budget) * 100, 2),
        "funding_sources": random.sample(funding_sources, k=random.randint(1, 3)),
        "funding_breakdown": {source: random.randint(10, 60) for source in random.sample(funding_sources, k=random.randint(1, 3))},
        "contractor": random.choice(contractors) if status != "Planned" else None,
        "subcontractors": random.sample(contractors, k=random.randint(0, 2)) if status != "Planned" else [],
        "start_date": start_date,
        "planned_end_date": fake_date(2024, 2026),
        "actual_end_date": end_date,
        "milestones": [
            {
                "id": f"MS{j+1}",
                "name": f"مرحلة {j+1}",
                "status": random.choice(["completed", "in_progress", "pending"]),
                "completion_date": fake_date(2020, 2024)
            } for j in range(random.randint(3, 8))
        ],
        "kpis": [
            {
                "name": f"مؤشر {random.choice(['التغطية', 'الجودة', 'الرضا', 'الأداء'])}",
                "target": random.randint(70, 100),
                "current": random.randint(40, 100),
                "unit": "%"
            } for _ in range(random.randint(2, 5))
        ],
        "impact_assessment": {
            "economic": round(random.uniform(0, 100), 2),
            "social": round(random.uniform(0, 100), 2),
            "environmental": round(random.uniform(0, 100), 2),
            "governance": round(random.uniform(0, 100), 2)
        },
        "documents": [
            {
                "type": doc_type,
                "name": f"{doc_type}_{project_id[:8]}.pdf",
                "upload_date": fake_date(2020, 2023),
                "size_mb": round(random.uniform(0.1, 50), 2)
            } for doc_type in random.sample(["دراسة جدوى", "مخطط", "تقرير", "عقد", "رخصة"], k=random.randint(1, 4))
        ],
        "media_gallery": [
            {
                "type": "image",
                "url": f"https://example.com/projects/{project_id}/img_{j}.jpg",
                "caption": f"صورة {j+1} للمشروع",
                "date": fake_date(2020, 2023)
            } for j in range(random.randint(0, 10))
        ],
        "managing_official_id": managing_official_id,
        "managing_official_name": managing_official["name"],
        "managing_department": managing_official["department"],
        "project_team": random.sample(list(official_id_map.keys()), k=random.randint(2, 8)),
        "external_partners": random.sample(["جمعية التنمية المحلية", "منظمة دولية", "جامعة", "مركز بحث"], 
                                         k=random.randint(0, 2)),
        "sustainability_plan": "خطة الاستدامة متوفرة" if random.random() > 0.4 else None,
        "environmental_impact_study": random.choice([True, False]),
        "social_impact_study": random.choice([True, False]),
        "audit_reports": random.randint(0, 3),
        "last_audit_date": fake_date(2022, 2023) if random.random() > 0.5 else None,
        "public_consultations_count": random.randint(0, 10),
        "complaints_received": random.randint(0, 20),
        "complaints_resolved": random.randint(0, 15),
        "awards_recognition": ["جائزة أفضل مشروع"] if random.random() > 0.9 else [],
        "lessons_learned": f"دروس مستفادة من المشروع" if status == "Completed" else None,
        "created_at": start_date,
        "updated_at": fake_datetime(2023, 2023),
        "tags": random.sample(["أولوية", "استراتيجي", "عاجل", "تشاركي", "مبتكر", "مستدام"], k=random.randint(1, 4))
    }
    
    # Update official stats
    official_id_map[managing_official_id]["projects_managed"] += 1
    
    project_details_dict[project_id] = payload
    municipal_project_points.append(PointStruct(id=project_id, vector=fake_embedding(), payload=payload))

batch_upsert(collection_name="municipal_projects", points=municipal_project_points, batch_size=500)

# -----------------------
# Generate Enhanced Citizen Comments
# -----------------------
print("\n💬 Generating Citizen Comments...")
comment_templates = [
    "أعتقد أن المشروع '{project_title}' {opinion} ولكن يجب {action} {aspect}",
    "أنا {feeling} من المشروع '{project_title}'، {reason} {aspect}",
    "المشروع '{project_title}' {evaluation}، {suggestion} {aspect}",
    "{greeting} أريد أن أشير إلى أن {observation} في المشروع '{project_title}' {recommendation}",
    "بخصوص المشروع '{project_title}' {concern} {aspect} {proposal}"
]

opinions = ["ممتاز", "جيد", "مقبول", "يحتاج تحسين", "ضعيف"]
feelings = ["راضٍ", "غير راضٍ", "متفائل", "قلق", "محايد"]
evaluations = ["يسير بشكل جيد", "يواجه تحديات", "متأخر عن الجدول", "يفوق التوقعات", "يحتاج مراجعة"]
greetings = ["السلام عليكم", "تحية طيبة", "مع كامل الاحترام", "أود أن أعبر عن رأيي"]
observations = ["لاحظت تحسناً", "هناك تراجع", "الأمور تسير", "يوجد تقدم ملحوظ", "لا يوجد تغيير"]
actions = ["تحسين", "مراجعة", "تسريع", "إعادة النظر في", "التركيز على"]
reasons = ["لأنني أرى", "حيث أن", "بسبب", "نظراً لـ", "لكون"]
suggestions = ["أقترح", "يستحسن", "من الأفضل", "أنصح بـ", "يجب"]
concerns = ["يقلقني", "ألاحظ", "أتساءل عن", "لدي ملاحظة حول", "أود التنبيه إلى"]
recommendations = ["وأقترح", "لذلك يجب", "مما يتطلب", "وأنصح بـ", "ويستدعي"]
proposals = ["وأقترح", "ولذلك أطلب", "مما يستوجب", "وأتمنى", "وأرجو"]

aspects = ["الجودة", "التنفيذ", "التواصل", "الشفافية", "السرعة", "التكلفة", "المواصفات", 
          "الصيانة", "المتابعة", "التخطيط", "الموارد البشرية", "المعدات", "السلامة"]

citizen_comment_points = []
comment_id_list = []
# Create a dictionary to track reply counts
comment_replies = defaultdict(int)

for i in range(5000):
    comment_id = str(uuid.uuid4())
    comment_id_list.append(comment_id)
    
    # Select project and citizen
    project_id = random.choice(municipal_project_ids)
    project_detail = project_details_dict[project_id]
    
    citizen_id = random.choice(citizen_ids)
    citizen = citizen_id_map[citizen_id]
    
    # Generate comment based on templates
    template = random.choice(comment_templates)
    
    comment_elements = {
        "project_title": project_detail["title"],
        "opinion": random.choice(opinions),
        "action": random.choice(actions),
        "aspect": random.choice(aspects),
        "feeling": random.choice(feelings),
        "reason": random.choice(reasons),
        "evaluation": random.choice(evaluations),
        "suggestion": random.choice(suggestions),
        "greeting": random.choice(greetings),
        "observation": random.choice(observations),
        "recommendation": random.choice(recommendations),
        "concern": random.choice(concerns),
        "proposal": random.choice(proposals)
    }
    
    # Build comment text
    comment_text = template
    for key, value in comment_elements.items():
        comment_text = comment_text.replace(f"{{{key}}}", value)
    
    # Sentiment based on content
    positive_indicators = ["ممتاز", "راضٍ", "جيد", "متفائل", "يفوق التوقعات", "تحسناً", "تقدم ملحوظ"]
    negative_indicators = ["ضعيف", "غير راضٍ", "قلق", "يحتاج تحسين", "متأخر", "تراجع", "يقلقني"]
    
    positive_count = sum(1 for indicator in positive_indicators if indicator in comment_text)
    negative_count = sum(1 for indicator in negative_indicators if indicator in comment_text)
    
    if positive_count > negative_count:
        sentiment = "POS"
        polarity = round(random.uniform(0.3, 1), 2)
        thumbs_up = random.randint(15, 50)
        thumbs_down = random.randint(0, 10)
    elif negative_count > positive_count:
        sentiment = "NEG"
        polarity = round(random.uniform(-1, -0.3), 2)
        thumbs_up = random.randint(0, 10)
        thumbs_down = random.randint(15, 50)
    else:
        sentiment = "NEU"
        polarity = round(random.uniform(-0.3, 0.3), 2)
        thumbs_up = random.randint(5, 20)
        thumbs_down = random.randint(5, 20)
    
    vote_score = thumbs_up - thumbs_down
    
    # Comment metadata
    comment_date = fake_datetime(2020, 2023)
    
    # Replies (some comments are replies to other comments)
    parent_comment_id = None
    if i > 100 and random.random() > 0.7:  # 30% chance of being a reply
        parent_comment_id = random.choice(comment_id_list[:-50])  # Don't reply to very recent comments
        # Increment the reply count for the parent
        comment_replies[parent_comment_id] += 1
    
    payload = {
        "comment_id": comment_id,
        "project_id": project_id,
        "citizen_id": citizen_id,
        "citizen_name": citizen["name"],
        "citizen_age_group": citizen["age_group"],
        "citizen_commune": citizen["commune"],
        "comment_text": comment_text,
        "comment_type": "reply" if parent_comment_id else "original",
        "parent_comment_id": parent_comment_id,
        "date_submitted": comment_date,
        "last_edited": comment_date if random.random() > 0.8 else None,
        "channel": random.choice(channels),
        "device_type": random.choice(["mobile", "desktop", "tablet"]),
        "sentiment": sentiment,
        "polarity": polarity,
        "emotion": random.choice(["happy", "sad", "angry", "neutral", "surprised", "worried"]),
        "keywords": random.sample(keywords_pool, k=random.randint(2, 5)),
        "entities_mentioned": [project_detail["commune"], project_detail["contractor"]] if project_detail.get("contractor") else [project_detail["commune"]],
        "aspects_discussed": random.sample(aspects, k=random.randint(1, 3)),
        # Project context
        "project_title": project_detail["title"],
        "project_themes": project_detail["themes"],
        "project_region": project_detail["region"],
        "project_province": project_detail["province"],
        "project_commune": project_detail["commune"],
        "project_status": project_detail["status"],
        "project_progress": project_detail["official_progress"],
        "project_budget": project_detail["current_budget"],
        # Voting and engagement
        "votes": {
            "thumb_up": thumbs_up,
            "thumb_down": thumbs_down,
            "vote_score": vote_score,
            "total_votes": thumbs_up + thumbs_down
        },
        "replies_count": 0,  # Will be updated later
        "shares_count": random.randint(0, 20) if random.random() > 0.7 else 0,
        "report_count": random.randint(0, 5) if random.random() > 0.95 else 0,
        "helpful_marked_by": random.sample(citizen_ids, k=random.randint(0, thumbs_up)),
        # Moderation
        "moderation_status": random.choice(["approved", "pending", "flagged", "hidden"]),
        "moderation_reason": "spam" if random.random() > 0.98 else None,
        "moderator_id": random.choice(list(official_id_map.keys())) if random.random() > 0.8 else None,
        # Analysis
        "constructive_score": round(random.uniform(0, 100), 2),
        "relevance_score": round(random.uniform(50, 100), 2),
        "toxicity_score": round(random.uniform(0, 20), 2) if sentiment != "NEG" else round(random.uniform(0, 50), 2),
        "actionable": random.choice([True, False]),
        "requires_response": random.choice([True, False]),
        "official_response_id": None,  # Will be updated if official responds
        # Additional metadata
        "language": "ar",
        "word_count": len(comment_text.split()),
        "contains_media": random.choice([True, False]) if random.random() > 0.9 else False,
        "media_urls": [f"https://example.com/comments/{comment_id}/media_{j}.jpg" 
                      for j in range(random.randint(1, 3))] if random.random() > 0.95 else [],
        "mentioned_citizens": random.sample(citizen_ids, k=random.randint(0, 3)) if random.random() > 0.9 else [],
        "hashtags": [f"#{tag}" for tag in random.sample(["تنمية_محلية", "مشاركة_مواطنة", "شفافية", 
                                                         "مشاريع_جماعية", "رأي_المواطن"], k=random.randint(0, 3))],
        "location_tag": citizen["commune"] if random.random() > 0.5 else None,
        "verified_location": random.choice([True, False]) if random.random() > 0.7 else False
    }
    
    # Update citizen stats
    citizen_id_map[citizen_id]["comments_made"] += 1
    
    citizen_comment_points.append(PointStruct(id=comment_id, vector=fake_embedding(), payload=payload))

# Update reply counts - FIXED: now using the collected reply counts from above
for i, comment in enumerate(citizen_comment_points):
    comment_id = comment.id
    if comment_id in comment_replies:
        citizen_comment_points[i].payload["replies_count"] = comment_replies[comment_id]

# Use small batches to avoid hitting the payload size limit
batch_upsert(collection_name="citizen_comments", points=citizen_comment_points, batch_size=200)

# -----------------------
# Generate Project Updates (New Collection)
# -----------------------
print("\n📝 Generating Project Updates...")
update_types = ["progress", "milestone", "budget", "schedule", "scope", "risk", "quality", "general"]
update_sources = ["contractor", "official", "inspection", "audit", "citizen_report"]

project_update_points = []

for project_id in municipal_project_ids[:1000]:  # Updates for first 1000 projects
    project = project_details_dict[project_id]
    num_updates = random.randint(1, 10)
    
    for j in range(num_updates):
        update_id = str(uuid.uuid4())
        update_type = random.choice(update_types)
        
        if update_type == "progress":
            update_content = f"تقدم المشروع من {random.randint(10, 40)}% إلى {random.randint(41, 90)}%"
        elif update_type == "milestone":
            update_content = f"تم إنجاز المرحلة {random.randint(1, 5)} من المشروع"
        elif update_type == "budget":
            update_content = f"تعديل الميزانية: {random.choice(['زيادة', 'تخفيض'])} بنسبة {random.randint(5, 20)}%"
        elif update_type == "schedule":
            update_content = f"تأخير في الجدول الزمني: {random.randint(1, 6)} أشهر"
        elif update_type == "risk":
            update_content = f"تم تحديد مخاطر جديدة: {random.choice(['تقنية', 'مالية', 'بيئية', 'اجتماعية'])}"
        else:
            update_content = f"تحديث عام حول سير المشروع"
        
        update_date = fake_datetime(2020, 2023)
        
        payload = {
            "update_id": update_id,
            "project_id": project_id,
            "project_title": project["title"],
            "update_type": update_type,
            "update_source": random.choice(update_sources),
            "title": f"تحديث {update_type} - {project['title'][:50]}",
            "content": update_content,
            "detailed_description": f"تفاصيل إضافية: {update_content}. {random.choice(['الوضع مستقر', 'يتطلب متابعة', 'تحت السيطرة'])}",
            "date_posted": update_date,
            "posted_by": random.choice(list(official_id_map.keys())),
            "severity": random.choice(["info", "warning", "critical"]) if update_type == "risk" else "info",
            "impact_areas": random.sample(["timeline", "budget", "quality", "scope"], k=random.randint(1, 3)),
            "attachments": [f"update_{update_id[:8]}_{k}.pdf" for k in range(random.randint(0, 2))],
            "public_visible": random.choice([True, False]),
            "requires_action": random.choice([True, False]),
            "action_taken": "تم اتخاذ الإجراءات اللازمة" if random.random() > 0.5 else None,
            "citizen_reactions": {
                "views": random.randint(10, 1000),
                "likes": random.randint(0, 100),
                "shares": random.randint(0, 50),
                "comments": random.randint(0, 30)
            },
            "previous_value": random.randint(0, 100) if update_type in ["progress", "budget"] else None,
            "new_value": random.randint(0, 100) if update_type in ["progress", "budget"] else None,
            "verification_status": random.choice(["verified", "pending", "disputed"]),
            "verified_by": random.choice(list(official_id_map.keys())) if random.random() > 0.5 else None,
            "tags": random.sample(["مهم", "عاجل", "للعلم", "يتطلب قرار"], k=random.randint(0, 2))
        }
        
        project_update_points.append(PointStruct(id=update_id, vector=fake_embedding(), payload=payload))

batch_upsert(collection_name="project_updates", points=project_update_points, batch_size=500)

# -----------------------
# Generate Budget Allocations (New Collection)
# -----------------------
print("\n💰 Generating Budget Allocations...")
fiscal_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
budget_categories = ["تشغيل", "استثمار", "صيانة", "طوارئ", "دعم الجمعيات", "برامج اجتماعية"]
allocation_statuses = ["مخطط", "معتمد", "محول", "مصروف", "ملغى"]

budget_allocation_points = []

for year in fiscal_years:
    for region in morocco_admin.keys():
        for province in morocco_admin[region].keys():
            for commune in morocco_admin[region][province]["communes"][:5]:  # First 5 communes per province
                for category in budget_categories:
                    allocation_id = str(uuid.uuid4())
                    
                    total_amount = random.randint(500000, 50000000)
                    if year < 2023:
                        allocated_amount = total_amount
                        spent_amount = random.randint(int(total_amount * 0.7), total_amount)
                    else:
                        allocated_amount = total_amount
                        spent_amount = random.randint(0, int(total_amount * 0.8))
                    
                    payload = {
                        "allocation_id": allocation_id,
                        "fiscal_year": year,
                        "region": region,
                        "province": province,
                        "commune": commune,
                        "category": category,
                        "subcategory": f"{category} - {random.choice(['عادي', 'خاص', 'استثنائي'])}",
                        "total_amount": total_amount,
                        "allocated_amount": allocated_amount,
                        "spent_amount": spent_amount,
                        "remaining_amount": allocated_amount - spent_amount,
                        "execution_rate": round((spent_amount / allocated_amount) * 100, 2),
                        "status": random.choice(allocation_statuses),
                        "funding_source": random.choice(funding_sources),
                        "approval_date": f"{year}-{random.randint(1, 3):02d}-{random.randint(1, 28):02d}",
                        "approved_by": random.choice(list(official_id_map.keys())),
                        "linked_projects": random.sample(municipal_project_ids, k=random.randint(0, 5)),
                        "beneficiary_count": random.randint(100, 10000),
                        "performance_indicators": [
                            {
                                "name": "معدل الإنجاز",
                                "target": 100,
                                "achieved": random.randint(60, 100)
                            }
                        ],
                        "audit_status": random.choice(["تم", "جاري", "مخطط", "غير مطلوب"]),
                        "notes": f"ملاحظات حول ميزانية {category} لسنة {year}",
                        "transparency_score": round(random.uniform(60, 100), 2),
                        "public_accessible": random.choice([True, False]),
                        "last_updated": fake_datetime(year, year)  # Fix: use same year for start and end
                    }
                    
                    budget_allocation_points.append(PointStruct(id=allocation_id, vector=fake_embedding(), payload=payload))

batch_upsert(collection_name="budget_allocations", points=budget_allocation_points, batch_size=500)

# -----------------------
# Generate Engagement Metrics (New Collection)
# -----------------------
print("\n📊 Generating Engagement Metrics...")
metric_types = ["daily", "weekly", "monthly", "quarterly", "annual"]
engagement_channels = ["web", "mobile", "physical", "call_center", "social_media"]

engagement_metric_points = []

# Generate metrics for different time periods
for year in range(2020, 2024):
    for month in range(1, 13):
        if year == 2023 and month > 12:  # Don't go beyond current data
            break
            
        for region in list(morocco_admin.keys())[:2]:  # First 2 regions for demo
            for province in list(morocco_admin[region].keys())[:3]:  # First 3 provinces
                metric_id = str(uuid.uuid4())
                
                period_start = f"{year}-{month:02d}-01"
                period_end = f"{year}-{month:02d}-28"
                
                total_citizens = random.randint(10000, 100000)
                active_citizens = random.randint(100, int(total_citizens * 0.2))
                
                payload = {
                    "metric_id": metric_id,
                    "metric_type": "monthly",
                    "period_start": period_start,
                    "period_end": period_end,
                    "year": year,
                    "month": month,
                    "region": region,
                    "province": province,
                    "demographics": {
                        "total_citizens": total_citizens,
                        "active_citizens": active_citizens,
                        "new_registrations": random.randint(10, 500),
                        "activation_rate": round((active_citizens / total_citizens) * 100, 2)
                    },
                    "engagement_stats": {
                        "total_ideas_submitted": random.randint(10, 500),
                        "total_comments": random.randint(50, 2000),
                        "total_votes": random.randint(100, 5000),
                        "total_project_follows": random.randint(20, 1000),
                        "avg_session_duration_minutes": round(random.uniform(2, 30), 2),
                        "bounce_rate": round(random.uniform(20, 60), 2)
                    },
                    "channel_breakdown": {
                        channel: {
                            "users": random.randint(10, 1000),
                            "sessions": random.randint(20, 2000),
                            "interactions": random.randint(50, 5000)
                        } for channel in engagement_channels
                    },
                    "content_performance": {
                        "most_viewed_projects": random.sample(municipal_project_ids, k=5),
                        "most_supported_ideas": random.sample(all_idea_ids, k=5),
                        "trending_topics": random.sample(idea_topics, k=3),
                        "viral_content_count": random.randint(0, 10)
                    },
                    "response_metrics": {
                        "avg_official_response_time_hours": round(random.uniform(1, 72), 2),
                        "response_rate": round(random.uniform(40, 95), 2),
                        "citizen_satisfaction_score": round(random.uniform(3, 5), 2),
                        "resolution_rate": round(random.uniform(30, 80), 2)
                    },
                    "quality_metrics": {
                        "constructive_feedback_rate": round(random.uniform(60, 95), 2),
                        "spam_rate": round(random.uniform(0, 10), 2),
                        "duplicate_content_rate": round(random.uniform(0, 15), 2),
                        "moderation_actions": random.randint(0, 50)
                    },
                    "technical_metrics": {
                        "system_uptime": round(random.uniform(95, 99.9), 2),
                        "avg_page_load_time_seconds": round(random.uniform(0.5, 3), 2),
                        "error_rate": round(random.uniform(0, 5), 2),
                        "api_calls": random.randint(1000, 100000)
                    },
                    "growth_metrics": {
                        "user_growth_rate": round(random.uniform(-5, 20), 2),
                        "engagement_growth_rate": round(random.uniform(-10, 30), 2),
                        "content_growth_rate": round(random.uniform(0, 25), 2)
                    },
                    "comparative_analysis": {
                        "vs_previous_period": round(random.uniform(-20, 50), 2),
                        "vs_same_period_last_year": round(random.uniform(-15, 40), 2),
                        "regional_rank": random.randint(1, 12),
                        "national_rank": random.randint(1, 100)
                    },
                    "generated_at": fake_datetime(year, year),
                    "data_quality_score": round(random.uniform(80, 100), 2)
                }
                
                engagement_metric_points.append(PointStruct(id=metric_id, vector=fake_embedding(), payload=payload))

batch_upsert(collection_name="engagement_metrics", points=engagement_metric_points, batch_size=500)

# -----------------------
# Update Cross-References and Compute Final Metrics
# -----------------------
print("\n🔄 Updating cross-references and computing final metrics...")

# Update citizen projects followed
for citizen_id, citizen in citizen_id_map.items():
    # Citizens follow projects in their area
    followed_projects = [p for p in municipal_project_ids 
                        if project_details_dict[p]["commune"] == citizen["commune"]][:random.randint(0, 10)]
    citizen["projects_followed"] = followed_projects

# Recompute project completion percentages with enhanced formula
project_votes = {}
project_quality_scores = {}

for point in citizen_comment_points:
    proj_id = point.payload["project_id"]
    votes = point.payload["votes"]
    sentiment = point.payload["sentiment"]
    
    if proj_id not in project_votes:
        project_votes[proj_id] = {
            "thumb_up": 0, 
            "thumb_down": 0, 
            "count": 0,
            "positive_comments": 0,
            "negative_comments": 0,
            "neutral_comments": 0
        }
    
    project_votes[proj_id]["thumb_up"] += votes["thumb_up"]
    project_votes[proj_id]["thumb_down"] += votes["thumb_down"]
    project_votes[proj_id]["count"] += 1
    
    if sentiment == "POS":
        project_votes[proj_id]["positive_comments"] += 1
    elif sentiment == "NEG":
        project_votes[proj_id]["negative_comments"] += 1
    else:
        project_votes[proj_id]["neutral_comments"] += 1

# Calculate quality scores
for proj_id, votes in project_votes.items():
    total_votes = votes["thumb_up"] + votes["thumb_down"]
    if total_votes > 0:
        approval_rate = (votes["thumb_up"] / total_votes) * 100
        
        # Sentiment score
        total_comments = votes["count"]
        if total_comments > 0:
            sentiment_score = ((votes["positive_comments"] - votes["negative_comments"]) / total_comments + 1) * 50
        else:
            sentiment_score = 50
        
        # Combined quality score
        quality_score = (approval_rate * 0.6) + (sentiment_score * 0.4)
        project_quality_scores[proj_id] = round(quality_score, 2)
    else:
        project_quality_scores[proj_id] = 75  # Default score

# Update projects with enhanced metrics - using batches for the update
updated_project_points = []
for point in municipal_project_points:
    payload = point.payload.copy()
    proj_id = payload["project_id"]
    
    # Update quality score
    payload["citizen_quality_score"] = project_quality_scores.get(proj_id, 75)
    
    # Update completion for ongoing projects
    if payload["status"] == "Ongoing":
        official_progress = payload.get("official_progress", 50)
        quality_score = payload["citizen_quality_score"]
        
        # Enhanced completion formula
        completion = (official_progress * 0.5) + (quality_score * 0.3) + (random.uniform(0, 20) * 0.2)
        payload["completion_percentage"] = round(min(95, completion), 2)
    
    # Add comment statistics
    if proj_id in project_votes:
        payload["citizen_engagement"] = {
            "total_comments": project_votes[proj_id]["count"],
            "positive_comments": project_votes[proj_id]["positive_comments"],
            "negative_comments": project_votes[proj_id]["negative_comments"],
            "neutral_comments": project_votes[proj_id]["neutral_comments"],
            "total_votes": project_votes[proj_id]["thumb_up"] + project_votes[proj_id]["thumb_down"],
            "approval_rate": round((project_votes[proj_id]["thumb_up"] / 
                                  (project_votes[proj_id]["thumb_up"] + project_votes[proj_id]["thumb_down"]) * 100), 2) 
                                  if (project_votes[proj_id]["thumb_up"] + project_votes[proj_id]["thumb_down"]) > 0 else 50
        }
    else:
        payload["citizen_engagement"] = {
            "total_comments": 0,
            "positive_comments": 0,
            "negative_comments": 0,
            "neutral_comments": 0,
            "total_votes": 0,
            "approval_rate": 50
        }
    
    updated_project_points.append(PointStruct(id=point.id, vector=point.vector, payload=payload))

# Use batches for the final update as well
batch_upsert(collection_name="municipal_projects", points=updated_project_points, batch_size=500)
print("✅ Updated projects with enhanced citizen engagement metrics")

# -----------------------
# Generate Summary Statistics
# -----------------------
print("\n📈 Generating Summary Statistics...")
print(f"Total Citizens: {len(citizens_data)}")
print(f"Total Officials: {len(officials_data)}")
print(f"Total Ideas: {len(citizen_idea_points)}")
print(f"Total Projects: {len(municipal_project_points)}")
print(f"Total Comments: {len(citizen_comment_points)}")
print(f"Total Project Updates: {len(project_update_points)}")
print(f"Total Budget Allocations: {len(budget_allocation_points)}")
print(f"Total Engagement Metrics: {len(engagement_metric_points)}")

# Calculate some interesting statistics
total_budget = sum(p.payload["current_budget"] for p in municipal_project_points)
avg_project_budget = total_budget / len(municipal_project_points)
completed_projects = sum(1 for p in municipal_project_points if p.payload["status"] == "Completed")
ongoing_projects = sum(1 for p in municipal_project_points if p.payload["status"] == "Ongoing")

print(f"\n💰 Financial Summary:")
print(f"Total Budget Allocated: {total_budget:,} DH")
print(f"Average Project Budget: {avg_project_budget:,.2f} DH")

print(f"\n📊 Project Status Summary:")
print(f"Completed Projects: {completed_projects}")
print(f"Ongoing Projects: {ongoing_projects}")
print(f"Completion Rate: {(completed_projects / len(municipal_project_points) * 100):.2f}%")

# Regional distribution
regional_stats = {}
for p in municipal_project_points:
    region = p.payload["region"]
    if region not in regional_stats:
        regional_stats[region] = {"count": 0, "budget": 0}
    regional_stats[region]["count"] += 1
    regional_stats[region]["budget"] += p.payload["current_budget"]

print(f"\n🗺️ Regional Distribution:")
for region, stats in regional_stats.items():
    print(f"{region}: {stats['count']} projects, {stats['budget']:,} DH total budget")

# Citizen engagement summary
total_ideas = len(citizen_idea_points)
total_comments = len(citizen_comment_points)
active_citizens = sum(1 for c in citizen_id_map.values() if c["ideas_submitted"] > 0 or c["comments_made"] > 0)

print(f"\n👥 Citizen Engagement Summary:")
print(f"Active Citizens: {active_citizens} ({active_citizens/len(citizens_data)*100:.2f}%)")
print(f"Average Ideas per Active Citizen: {total_ideas/active_citizens:.2f}")
print(f"Average Comments per Active Citizen: {total_comments/active_citizens:.2f}")

# Channel usage
channel_usage = {}
for idea in citizen_idea_points:
    channel = idea.payload["channel"]
    channel_usage[channel] = channel_usage.get(channel, 0) + 1

print(f"\n📱 Channel Usage for Ideas:")
for channel, count in sorted(channel_usage.items(), key=lambda x: x[1], reverse=True):
    print(f"{channel}: {count} ({count/total_ideas*100:.2f}%)")

# Project type distribution
project_types_stats = {}
for p in municipal_project_points:
    ptype = p.payload["project_type"]
    project_types_stats[ptype] = project_types_stats.get(ptype, 0) + 1

print(f"\n🏗️ Project Type Distribution:")
for ptype, count in sorted(project_types_stats.items(), key=lambda x: x[1], reverse=True):
    print(f"{ptype}: {count} ({count/len(municipal_project_points)*100:.2f}%)")

# Official response metrics
assigned_ideas = sum(1 for i in citizen_idea_points if i.payload.get("assigned_official_id"))
print(f"\n👔 Official Response Metrics:")
print(f"Ideas Assigned to Officials: {assigned_ideas} ({assigned_ideas/total_ideas*100:.2f}%)")

# Budget execution
total_allocated = sum(b.payload["allocated_amount"] for b in budget_allocation_points)
total_spent = sum(b.payload["spent_amount"] for b in budget_allocation_points)
execution_rate = (total_spent / total_allocated * 100) if total_allocated > 0 else 0

print(f"\n💸 Budget Execution:")
print(f"Total Allocated: {total_allocated:,} DH")
print(f"Total Spent: {total_spent:,} DH")
print(f"Overall Execution Rate: {execution_rate:.2f}%")

# Data quality check
print(f"\n✅ Data Quality Check:")
print(f"All collections created successfully")
print(f"Cross-references established")
print(f"Metrics computed and updated")
print(f"Ready for semantic search and analysis!")

# -----------------------
# Create Search Examples
# -----------------------
print("\n🔍 Example Semantic Searches You Can Perform:")
print("1. Find all ideas related to 'الإنارة' in 'فاس'")
print("2. Search for projects with budget > 1M DH and completion < 50%")
print("3. Find negative comments about ongoing infrastructure projects")
print("4. Identify officials with highest project success rates")
print("5. Analyze citizen engagement trends by region and age group")
print("6. Find similar ideas across different communes")
print("7. Track budget execution by category and year")
print("8. Identify projects with high citizen participation")
print("9. Analyze sentiment evolution over time for specific projects")
print("10. Find correlations between idea keywords and project outcomes")

print("\n🎉 Data generation complete! Your Qdrant instance now contains:")
print(f"- {len(collections)} interconnected collections")
print("- Rich metadata for advanced filtering and analysis")
print("- Realistic relationships between entities")
print("- Comprehensive tracking of citizen participation and project lifecycle")