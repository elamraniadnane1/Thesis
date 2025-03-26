from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.models import PointStruct
import uuid
import random
import datetime

# Connect to local Qdrant instance
qdrant = QdrantClient(host="localhost", port=6333)

# -----------------------
# Create Collections if They Don't Exist
# -----------------------
citizen_idea_collection = "citizen_ideas"
if not qdrant.collection_exists(citizen_idea_collection):
    qdrant.create_collection(
        collection_name=citizen_idea_collection,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("✅ Created collection:", citizen_idea_collection)

municipal_project_collection = "municipal_projects"
if not qdrant.collection_exists(municipal_project_collection):
    qdrant.create_collection(
        collection_name=municipal_project_collection,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("✅ Created collection:", municipal_project_collection)

citizen_comments_collection = "citizen_comments"
if not qdrant.collection_exists(citizen_comments_collection):
    qdrant.create_collection(
        collection_name=citizen_comments_collection,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("✅ Created collection:", citizen_comments_collection)

# -----------------------
# Define Morocco Administrative Divisions
# -----------------------
morocco_admin = {
    "جهة طنجة-تطوان-الحسيمة": {
        "عمالة طنجة-أصيلة": ["طنجة", "أصيلة", "اكزناية", "المنزلة", "دار الشاوي", "الزميج", "الساحل الشمالي", "أحد الغربية", "سوق الداخل"],
        "عمالة المضيق-الفنيدق": ["المضيق", "الفنيدق", "مرتيل", "العليين", "بليونش"],
        "إقليم تطوان": ["تطوان", "الحمراء", "بن قريش", "الملاليين", "أزلا", "وادي لو", "عين لحصن", "الخروب", "بني سعيد"],
        "إقليم الحسيمة": ["الحسيمة", "تارجيست", "بني بوعياش", "امزورن", "بني حذيفة", "بني عمارت", "كتامة", "إساكن", "تماسينت", "بني بوفراح"],
        "إقليم العرائش": ["العرائش", "القصر الكبير", "العوامرة", "ريصانة الجنوبية", "ريصانة الشمالية", "زعرورة", "تطفت", "بني كرفط", "سوق الطلبة"],
        "إقليم شفشاون": ["شفشاون", "باب برد", "باب تازة", "بني دركول", "بني فغلوم", "بني رزين", "بني سلمان", "تاموروت", "تنقوب", "واد ملحة"],
        "إقليم وزان": ["وزان", "المجاعرة", "سيدي بوصبر", "سيدي رضوان", "زومي", "عين دريج", "بوقرة", "سيدي أحمد الشريف", "سيدي امحمد الشريف", "سيدي عبد الله"],
        "إقليم فحص-أنجرة": ["أنجرة", "الفحص", "قصر المجاز", "ملوسة", "جبل الحبيب", "تاغرامت", "الخروب", "دار الشاوي", "الزميج", "الساحل الشمالي"]
    },
    "جهة الشرق": {
        "عمالة وجدة-أنكاد": ["وجدة", "بني درار", "عين الصفا", "سيدي موسى لمهاية", "اسلي", "مستفركي", "نعيمة", "بني خالد", "لبصارة"],
        "إقليم الناظور": ["الناظور", "زايو", "العروي", "سلوان", "بني انصار", "فرخانة", "أزغنغان", "إحدادن", "بني سيدال الجبل", "بني سيدال لوطا"],
        "إقليم بركان": ["بركان", "أحفير", "السعيدية", "سيدي سليمان شراعة", "مداغ", "عين الركادة", "زكزل", "فزوان", "الشويحية", "لمريس"],
        "إقليم تاوريرت": ["تاوريرت", "العنصر", "العيون سيدي ملوك", "مستكمار", "سيدي لحسن", "ملقى الويدان", "دبدو", "سيدي علي بلقاسم", "تنشرفي", "مستكمر"],
        "إقليم جرادة": ["جرادة", "عين بني مطهر", "تويسيت", "لعوينات", "راس عصفور", "كفايت", "لمريجة", "بني مطهر", "بني يعلى", "بني كيل"],
        "إقليم فجيج": ["بوعرفة", "فجيج", "بني تجيت", "تندرارة", "معتركة", "بني كيل", "بني مطهر", "بني يعلى", "بني وريمش"],
        "إقليم الدريوش": ["الدريوش", "ميضار", "بن الطيب", "دار الكبداني", "تمسمان", "بودينار", "كرونة", "تزاغين", "أولاد بوبكر", "أمجاو"],
        "إقليم جرسيف": ["جرسيف", "تادرت", "صاكة", "لمريجة", "هوارة أولاد رحو", "مزكيتام", "بركين", "بني مقبل", "بني وريمش"]
    },
    "جهة فاس-مكناس": {
        "عمالة فاس": ["فاس", "زواغة", "سايس", "أكدال", "جنان الورد", "مرنيسة", "مولاي يعقوب", "عين الله", "عين الشقف", "سبع رواضي"],
        "عمالة مكناس": ["مكناس", "حمرية", "المنصور", "الإسماعيلية", "زرهون", "عين عرمة", "عين كرمة", "مجاط", "ويسلان", "سيدي سليمان"],
        "إقليم صفرو": ["صفرو", "البهاليل", "رباط الخير", "أدرج", "أغبالو أقورار", "أولاد مكودو", "عين تمكناي", "الدار الحمراء", "إغزران", "سيدي يوسف بن أحمد"],
        "إقليم إفران": ["إفران", "أزرو", "مشليفن", "ضاية عوا", "عين اللوح", "بن صميم", "تيزكيت", "سيدي المخفي", "تيمحضيت", "واد إفران"],
        "إقليم الحاجب": ["الحاجب", "عين تاوجطات", "سبع عيون", "أكوراي", "أيت بوبيدمان", "أيت يعزم", "إقدار", "سيدي داود", "تامشاشاط", "تازوطة"],
        "إقليم بولمان": ["ميسور", "أوطاط الحاج", "المرس", "تالسينت", "إنجيل", "فريطيسة", "سكورة", "سيدي بوطيب", "ألميس مرموشة", "بني كيل"],
        "إقليم تاونات": ["تاونات", "تيسة", "قرية با محمد", "غفساي", "الوردزاغ", "عين عائشة", "عين مديونة", "بني وليد", "بوهودة", "مولاي بوشتى"],
        "إقليم تازة": ["تازة", "واد أمليل", "تايناست", "أكنول", "مطماطة", "باب مرزوقة", "بني فراسن", "بني لنت", "بني فتاح", "بني ورين"]
    },
    "جهة الرباط-سلا-قنيطرة": {
        "عمالة الرباط": ["الرباط", "سيدي موسى", "الحي الحديث", "الحي القديم", "الرياض"],
        "عمالة سلا": ["سلا", "البساتين", "الشباب", "العربي"],
        "إقليم قنيطرة": ["قنيطرة", "دار البيضاء الجديدة", "القصر", "المجمع"]
    },
    "جهة بني ملال-خنيفرة": {
        "عمالة بني ملال": ["بني ملال", "المحمدية", "الحسيمة"],
        "عمالة خنيفرة": ["خنيفرة", "أزيلال", "صغير خنيفرة"],
        "إقليم فقيه بن صالح": ["فقيه بن صالح", "المجلس"]
    },
    "جهة الدار البيضاء-سطات": {
        "عمالة الدار البيضاء": ["الدار البيضاء", "المحمدية", "الوسط", "البرج"],
        "عمالة سطات": ["سطات", "دار التبانة", "المنارة", "المعين"]
    },
    "جهة مراكش-آسفي": {
        "عمالة مراكش": ["مراكش", "الحمراء", "الصالحية", "النعمانية", "الكتبية"],
        "عمالة آسفي": ["آسفي", "الوسط", "البوادي"],
        "إقليم سافي": ["سافي", "الكويرة", "أولاد دراج"]
    },
    "جهة درعة-تافيلالت": {
        "عمالة ورزازات": ["ورزازات", "أوزود", "أكواد", "الورزازات الوسطى"],
        "عمالة تنغير": ["تنغير", "تنجير", "تنغير الجنوبية"],
        "إقليم زاغورة": ["زاغورة", "إرراشيد", "الزاغورة الكبرى"]
    },
    "جهة سوس ماسة": {
        "عمالة أكادير": ["أكادير", "تامسنا", "تامسوس", "أكادير الوسطى"],
        "عمالة تارودانت": ["تارودانت", "تيميتين", "العقيلة", "تارودانت القديمة"],
        "عمالة أولاد تيلا": ["أولاد تيلا", "بو تيس", "أولاد تيلا الوسطى"]
    },
    "جهة كلميم-وادي نون": {
        "عمالة جويلم": ["جويلم", "تان-تان", "جويلم الوسطى"],
        "عمالة سيدي إفني": ["سيدي إفني", "المردة", "سيدي إفني العليا"],
        "عمالة أوسرد": ["أوسرد", "أوسرد الصغيرة", "أوسرد الكبرى"]
    },
    "جهة العيون-الساقية الحمراء": {
        "عمالة العيون": ["العيون", "تارفايا", "الشاطئ", "العيون الوسطى"],
        "عمالة الدكلة": ["الدكلة", "سمارة", "الدكلة العليا"]
    },
    "جهة الداخلة-واد الذهب": {
        "عمالة دخلة": ["دخلة", "أولاد", "دخلة الوسطى"]
    }
}
  
def select_location():
    region = random.choice(list(morocco_admin.keys()))
    province = random.choice(list(morocco_admin[region].keys()))
    commune = random.choice(morocco_admin[region][province])
    return region, province, commune

def fake_embedding():
    return [random.uniform(-1, 1) for _ in range(384)]

def fake_date():
    start_date = datetime.datetime(2018, 1, 1)
    end_date = datetime.datetime(2023, 12, 31)
    random_seconds = random.randint(0, int((end_date - start_date).total_seconds()))
    random_date = start_date + datetime.timedelta(seconds=random_seconds)
    return random_date.strftime("%Y-%m-%d")

# Predefined additional lists
axes = ["النقل والإنارة", "النظافة والبيئة", "تنمية اجتماعية", "تنمية اقتصادية", "الرقمنة", "الصحة العامة"]
challenge_templates = [
    "تعاني المنطقة من {}.",
    "هناك مشكلة في {} تؤثر على السكان.",
    "تواجه الجماعة {} بشكل مستمر.",
    "نقص في {} يؤدي إلى مشاكل أمنية."
]
solution_templates = [
    "اقتراح بتفعيل {}.",
    "تنفيذ {} لتحسين الوضع.",
    "اقتراح إنشاء {} لمواجهة التحديات.",
    "تطبيق {} كحل لهذه المشكلة."
]
idea_topics = ["البنية التحتية", "الخدمات الاجتماعية", "الأمن", "الشفافية", "البيئة"]
keywords_pool = ["إنارة", "شوارع", "أمان", "نظافة", "بيئة", "خدمات", "مشاركة", "تواصل", "تكنولوجيا"]
channels = ["اللقاء", "تطبيق", "استمارة", "SNS", "هاتف"]
sentiments = ["POS", "NEG", "NEU"]

# -----------------------
# Generate 1000 Fake Citizen Idea Points
# -----------------------
citizen_points = []
all_idea_ids = []
for _ in range(1000):
    idea_id = str(uuid.uuid4())
    all_idea_ids.append(idea_id)
    region, province, commune = select_location()
    axis = random.choice(axes)
    challenge_text = random.choice(challenge_templates).format(random.choice(keywords_pool))
    solution_text = random.choice(solution_templates).format(random.choice(keywords_pool))
    sentiment = random.choice(sentiments)
    polarity = round(random.uniform(-1, 1), 2)
    topic = random.choice(idea_topics)
    keywords = random.sample(keywords_pool, k=random.randint(2, 4))
    
    payload = {
        "idea_id": idea_id,
        "axis": axis,
        "challenge": challenge_text,
        "solution": solution_text,
        "city": commune,  # using commune as city indicator for simplicity
        "commune": commune,
        "province": province,
        "CT": region,
        "channel": random.choice(channels),
        "date_submitted": fake_date(),
        "linked_project_ids": [],  # to be filled when projects are created
        "sentiment": sentiment,
        "polarity": polarity,
        "topic": topic,
        "keywords": keywords
    }
    citizen_points.append(PointStruct(id=idea_id, vector=fake_embedding(), payload=payload))
qdrant.upsert(collection_name=citizen_idea_collection, points=citizen_points)
print("✅ Inserted 1000 citizen idea points.")

# -----------------------
# Generate 1000 Fake Municipal Project Points
# -----------------------
project_statuses = ["Planned", "Ongoing", "Completed"]
participation_flags = [True, False]
funders_pool = ["صندوق التجهيز الجماعي", "وزارة الداخلية", "الشراكة المحلية"]
kpi_templates = ["تغطية {} %", "رفع مستوى {}"]
impact_templates = ["تحسين {}", "رفع مستوى {}"]

municipal_points = []
municipal_project_ids = []  # Save IDs for linking in comments
project_details_dict = {}

# When creating projects, we randomly select one or more idea IDs to link.
for _ in range(1000):
    project_id = str(uuid.uuid4())
    municipal_project_ids.append(project_id)
    region, province, commune = select_location()
    neighborhood = f"حي {random.randint(1, 20)}"
    title = f"مشروع {random.choice(idea_topics)} في {commune}"
    themes = f"تحسين {random.choice(idea_topics)}"
    status = random.choice(project_statuses)
    if status == "Ongoing":
        official_progress = random.randint(20, 90)
    elif status == "Planned":
        official_progress = 0
    else:
        official_progress = 100
    citizen_participation = random.choice(participation_flags)
    # Link between projects and ideas: randomly choose 1 to 3 idea IDs from citizen ideas
    linked_idea_ids = random.sample(all_idea_ids, k=random.randint(1, 3))
    budget = random.randint(50000, 1000000)
    funders = random.choice(funders_pool)
    kpi = random.choice(kpi_templates).format(random.choice(["الخدمات", "الأمان", "الإنارة"]))
    impact = random.choice(impact_templates).format(random.choice(["الأمان", "الخدمات", "البيئة", "الشفافية"]))
    
    payload = {
        "project_id": project_id,
        "title": title,
        "themes": themes,
        "CT": region,
        "province": province,
        "commune": commune,
        "neighborhood": neighborhood,
        "status": status,
        "official_progress": official_progress,
        "citizen_participation": citizen_participation,
        "linked_idea_ids": linked_idea_ids,  # Linked idea IDs for this project
        "budget": budget,
        "funders": funders,
        "kpi": kpi,
        "impact": impact
    }
    project_details_dict[project_id] = payload
    municipal_points.append(PointStruct(id=project_id, vector=fake_embedding(), payload=payload))
qdrant.upsert(collection_name=municipal_project_collection, points=municipal_points)
print("✅ Inserted 1000 municipal project points.")

# -----------------------
# Generate 1000 Fake Citizen Comment Points with Voting and Project Linkage
# -----------------------
citizen_names = [
    "أمين", "فاطمة", "سعيد", "مريم", "حسن", "ليلى", "يوسف", "سلوى", "علي", "نجلاء",
    "كريم", "خديجة", "محمد", "أسماء", "إبراهيم", "زينب", "سامي", "هدى", "وليد", "سميرة"
]
comment_templates = [
    "أعتقد أن المشروع '{project_title}' جيد ولكن يجب تحسين {aspect}.",
    "أنا راضٍ عن المشروع '{project_title}'، لكن هناك مشكلة في {aspect}.",
    "المشروع '{project_title}' ممتاز، ولكنه يحتاج إلى المزيد من {aspect}.",
    "اقتراحي هو تعديل {aspect} في المشروع '{project_title}'."
]
aspects = ["الصيانة", "التنفيذ", "الجودة", "التواصل", "التمويل", "الشفافية"]

citizen_comment_points = []
# For each comment, link it to a municipal project and simulate logical voting.
for _ in range(1000):
    comment_id = str(uuid.uuid4())
    project_id = random.choice(municipal_project_ids)
    project_detail = project_details_dict[project_id]
    citizen_name = random.choice(citizen_names)
    aspect = random.choice(aspects)
    template = random.choice(comment_templates)
    comment_text = template.format(project_title=project_detail["title"], aspect=aspect)
    sentiment = random.choice(sentiments)
    if sentiment == "POS":
        thumbs_up = random.randint(10, 30)
        thumbs_down = random.randint(0, 5)
    elif sentiment == "NEG":
        thumbs_up = random.randint(0, 5)
        thumbs_down = random.randint(10, 30)
    else:
        thumbs_up = random.randint(5, 15)
        thumbs_down = random.randint(5, 15)
    vote_score = thumbs_up - thumbs_down
    polarity = round(random.uniform(-1, 1), 2)
    
    payload = {
        "comment_id": comment_id,
        "project_id": project_id,
        "citizen_name": citizen_name,
        "comment_text": comment_text,
        "date_submitted": fake_date(),
        "channel": random.choice(channels),
        "sentiment": sentiment,
        "polarity": polarity,
        "keywords": random.sample(keywords_pool, k=random.randint(2, 4)),
        # Include linked project details for full context
        "project_title": project_detail["title"],
        "project_themes": project_detail["themes"],
        "project_CT": project_detail["CT"],
        "project_province": project_detail["province"],
        "project_commune": project_detail["commune"],
        "project_status": project_detail["status"],
        # Voting details
        "votes": {
            "thumb_up": thumbs_up,
            "thumb_down": thumbs_down,
            "vote_score": vote_score
        }
    }
    citizen_comment_points.append(PointStruct(id=comment_id, vector=fake_embedding(), payload=payload))
qdrant.upsert(collection_name=citizen_comments_collection, points=citizen_comment_points)
print("✅ Inserted 1000 citizen comment points.")

# -----------------------
# Compute Completion Percentage for Each Ongoing Project
# -----------------------
# For ongoing projects, compute:
#   completion = 0.6 * official_progress + 0.4 * citizen_approval_rate
# where citizen_approval_rate = (total thumbs_up / total votes)*100, averaged over comments for that project.
project_votes = {}
for point in citizen_comment_points:
    proj_id = point.payload["project_id"]
    votes = point.payload["votes"]
    if proj_id not in project_votes:
        project_votes[proj_id] = {"thumb_up": 0, "thumb_down": 0, "count": 0}
    project_votes[proj_id]["thumb_up"] += votes["thumb_up"]
    project_votes[proj_id]["thumb_down"] += votes["thumb_down"]
    project_votes[proj_id]["count"] += 1

updated_project_points = []
for point in municipal_points:
    payload = point.payload.copy()
    if payload["status"] == "Ongoing":
        official_progress = payload.get("official_progress", random.randint(20, 90))
        votes = project_votes.get(payload["project_id"], {"thumb_up": 0, "thumb_down": 0, "count": 0})
        total_votes = votes["thumb_up"] + votes["thumb_down"]
        if total_votes > 0:
            citizen_approval_rate = (votes["thumb_up"] / total_votes) * 100
        else:
            citizen_approval_rate = official_progress
        completion_percentage = 0.6 * official_progress + 0.4 * citizen_approval_rate
        payload["completion_percentage"] = round(completion_percentage, 2)
    else:
        payload["completion_percentage"] = 0 if payload["status"] == "Planned" else 100
    updated_project_points.append(PointStruct(id=point.id, vector=point.vector, payload=payload))

qdrant.upsert(collection_name=municipal_project_collection, points=updated_project_points)
print("✅ Updated ongoing projects with completion percentages.")
