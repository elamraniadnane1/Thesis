import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style for better visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Project timeline
start_date = datetime(2024, 9, 9)
end_date = datetime(2025, 7, 1)

# Define all project tasks with start dates, durations, and dependencies
tasks = [
    # Phase 1: Project Initiation & Discovery (Sep 9 - Oct 31, 2024)
    {
        'Task': 'Project Kickoff & Team Assembly',
        'Start': datetime(2024, 9, 9),
        'Duration': 7,
        'Category': 'Initiation',
        'Phase': 'Phase 1: Discovery'
    },
    {
        'Task': 'Civic Discovery & Citizen Needs Assessment',
        'Start': datetime(2024, 9, 16),
        'Duration': 21,
        'Category': 'Research',
        'Phase': 'Phase 1: Discovery'
    },
    {
        'Task': 'Privacy Landscape Mapping & CNDP Analysis',
        'Start': datetime(2024, 9, 16),
        'Duration': 21,
        'Category': 'Legal',
        'Phase': 'Phase 1: Discovery'
    },
    {
        'Task': 'Technical Feasibility & Integration Study',
        'Start': datetime(2024, 9, 23),
        'Duration': 14,
        'Category': 'Technical',
        'Phase': 'Phase 1: Discovery'
    },
    {
        'Task': 'REMACTO Integration Requirements Analysis',
        'Start': datetime(2024, 10, 7),
        'Duration': 14,
        'Category': 'Integration',
        'Phase': 'Phase 1: Discovery'
    },
    {
        'Task': 'Multilingual Requirements Definition (Arabic/French/Darija)',
        'Start': datetime(2024, 10, 14),
        'Duration': 10,
        'Category': 'Linguistic',
        'Phase': 'Phase 1: Discovery'
    },
    
    # Phase 2: Architecture & Design (Oct 21 - Dec 15, 2024)
    {
        'Task': 'Privacy-by-Design Architecture Planning',
        'Start': datetime(2024, 10, 21),
        'Duration': 21,
        'Category': 'Architecture',
        'Phase': 'Phase 2: Architecture'
    },
    {
        'Task': 'Microservices Architecture Design',
        'Start': datetime(2024, 10, 28),
        'Duration': 21,
        'Category': 'Architecture',
        'Phase': 'Phase 2: Architecture'
    },
    {
        'Task': 'Data Architecture & Privacy Controls Design',
        'Start': datetime(2024, 11, 4),
        'Duration': 21,
        'Category': 'Data',
        'Phase': 'Phase 2: Architecture'
    },
    {
        'Task': 'Security Architecture & Encryption Planning',
        'Start': datetime(2024, 11, 11),
        'Duration': 14,
        'Category': 'Security',
        'Phase': 'Phase 2: Architecture'
    },
    {
        'Task': 'AI/ML Pipeline Architecture Design',
        'Start': datetime(2024, 11, 18),
        'Duration': 21,
        'Category': 'AI/ML',
        'Phase': 'Phase 2: Architecture'
    },
    {
        'Task': 'Vector Database & RAG System Design',
        'Start': datetime(2024, 11, 25),
        'Duration': 14,
        'Category': 'AI/ML',
        'Phase': 'Phase 2: Architecture'
    },
    {
        'Task': 'API Specifications & Integration Interfaces',
        'Start': datetime(2024, 12, 2),
        'Duration': 14,
        'Category': 'Integration',
        'Phase': 'Phase 2: Architecture'
    },
    
    # Phase 3: Core Development (Dec 2, 2024 - Mar 31, 2025)
    {
        'Task': 'Development Environment Setup & CI/CD Pipeline',
        'Start': datetime(2024, 12, 2),
        'Duration': 14,
        'Category': 'Infrastructure',
        'Phase': 'Phase 3: Development'
    },
    {
        'Task': 'Privacy-Preserving Analytics Engine Development',
        'Start': datetime(2024, 12, 16),
        'Duration': 35,
        'Category': 'Core Development',
        'Phase': 'Phase 3: Development'
    },
    {
        'Task': 'Multilingual NLP Pipeline Implementation',
        'Start': datetime(2024, 12, 16),
        'Duration': 42,
        'Category': 'NLP',
        'Phase': 'Phase 3: Development'
    },
    {
        'Task': 'Sentiment Analysis Module (Arabic/French/Darija)',
        'Start': datetime(2024, 12, 23),
        'Duration': 28,
        'Category': 'NLP',
        'Phase': 'Phase 3: Development'
    },
    {
        'Task': 'Topic Modeling & Classification System',
        'Start': datetime(2025, 1, 6),
        'Duration': 28,
        'Category': 'NLP',
        'Phase': 'Phase 3: Development'
    },
    {
        'Task': 'Offensive Language Detection System',
        'Start': datetime(2025, 1, 13),
        'Duration': 21,
        'Category': 'NLP',
        'Phase': 'Phase 3: Development'
    },
    {
        'Task': 'Document Summarization Engine',
        'Start': datetime(2025, 1, 20),
        'Duration': 21,
        'Category': 'NLP',
        'Phase': 'Phase 3: Development'
    },
    {
        'Task': 'Vector Database Implementation & Optimization',
        'Start': datetime(2025, 1, 27),
        'Duration': 28,
        'Category': 'Database',
        'Phase': 'Phase 3: Development'
    },
    {
        'Task': 'RAG System Implementation & LLM Integration',
        'Start': datetime(2025, 2, 3),
        'Duration': 35,
        'Category': 'AI/ML',
        'Phase': 'Phase 3: Development'
    },
    {
        'Task': 'Differential Privacy Implementation',
        'Start': datetime(2025, 2, 10),
        'Duration': 21,
        'Category': 'Privacy',
        'Phase': 'Phase 3: Development'
    },
    {
        'Task': 'Federated Learning Architecture Implementation',
        'Start': datetime(2025, 2, 17),
        'Duration': 28,
        'Category': 'AI/ML',
        'Phase': 'Phase 3: Development'
    },
    {
        'Task': 'Homomorphic Encryption Integration',
        'Start': datetime(2025, 2, 24),
        'Duration': 21,
        'Category': 'Privacy',
        'Phase': 'Phase 3: Development'
    },
    
    # Phase 4: User Interface & Integration (Mar 3 - May 5, 2025)
    {
        'Task': 'Multilingual Web Interface Development',
        'Start': datetime(2025, 3, 3),
        'Duration': 35,
        'Category': 'Frontend',
        'Phase': 'Phase 4: UI/Integration'
    },
    {
        'Task': 'Mobile-First Responsive Design Implementation',
        'Start': datetime(2025, 3, 10),
        'Duration': 28,
        'Category': 'Frontend',
        'Phase': 'Phase 4: UI/Integration'
    },
    {
        'Task': 'Citizen Engagement Portal Development',
        'Start': datetime(2025, 3, 17),
        'Duration': 35,
        'Category': 'Frontend',
        'Phase': 'Phase 4: UI/Integration'
    },
    {
        'Task': 'Municipal Admin Dashboard Development',
        'Start': datetime(2025, 3, 24),
        'Duration': 28,
        'Category': 'Frontend',
        'Phase': 'Phase 4: UI/Integration'
    },
    {
        'Task': 'Privacy Dashboard & Citizen Rights Portal',
        'Start': datetime(2025, 3, 31),
        'Duration': 21,
        'Category': 'Privacy',
        'Phase': 'Phase 4: UI/Integration'
    },
    {
        'Task': 'REMACTO Platform Integration',
        'Start': datetime(2025, 4, 7),
        'Duration': 28,
        'Category': 'Integration',
        'Phase': 'Phase 4: UI/Integration'
    },
    {
        'Task': 'Legacy Municipal Systems Integration',
        'Start': datetime(2025, 4, 14),
        'Duration': 21,
        'Category': 'Integration',
        'Phase': 'Phase 4: UI/Integration'
    },
    {
        'Task': 'Voice Input System (WhatsApp/Phone Integration)',
        'Start': datetime(2025, 4, 21),
        'Duration': 14,
        'Category': 'Integration',
        'Phase': 'Phase 4: UI/Integration'
    },
    
    # Phase 5: Testing & Validation (Apr 7 - Jun 16, 2025)
    {
        'Task': 'Unit Testing & Code Quality Assurance',
        'Start': datetime(2025, 4, 7),
        'Duration': 35,
        'Category': 'Testing',
        'Phase': 'Phase 5: Testing'
    },
    {
        'Task': 'Privacy Compliance Testing & CNDP Validation',
        'Start': datetime(2025, 4, 14),
        'Duration': 28,
        'Category': 'Compliance',
        'Phase': 'Phase 5: Testing'
    },
    {
        'Task': 'Multilingual Testing & Cultural Validation',
        'Start': datetime(2025, 4, 21),
        'Duration': 21,
        'Category': 'Testing',
        'Phase': 'Phase 5: Testing'
    },
    {
        'Task': 'AI Model Performance & Bias Testing',
        'Start': datetime(2025, 4, 28),
        'Duration': 28,
        'Category': 'AI Testing',
        'Phase': 'Phase 5: Testing'
    },
    {
        'Task': 'Load Testing & Scalability Validation',
        'Start': datetime(2025, 5, 5),
        'Duration': 21,
        'Category': 'Performance',
        'Phase': 'Phase 5: Testing'
    },
    {
        'Task': 'Security Penetration Testing',
        'Start': datetime(2025, 5, 12),
        'Duration': 14,
        'Category': 'Security',
        'Phase': 'Phase 5: Testing'
    },
    {
        'Task': 'User Acceptance Testing with Citizens',
        'Start': datetime(2025, 5, 19),
        'Duration': 21,
        'Category': 'UAT',
        'Phase': 'Phase 5: Testing'
    },
    {
        'Task': 'Municipal Staff Training & Feedback Collection',
        'Start': datetime(2025, 5, 26),
        'Duration': 21,
        'Category': 'Training',
        'Phase': 'Phase 5: Testing'
    },
    
    # Phase 6: Deployment & Launch (May 26 - July 1, 2025)
    {
        'Task': 'Production Environment Setup',
        'Start': datetime(2025, 5, 26),
        'Duration': 14,
        'Category': 'Deployment',
        'Phase': 'Phase 6: Launch'
    },
    {
        'Task': 'Pilot Municipality Deployment (Oujda)',
        'Start': datetime(2025, 6, 2),
        'Duration': 14,
        'Category': 'Deployment',
        'Phase': 'Phase 6: Launch'
    },
    {
        'Task': 'System Monitoring & Performance Optimization',
        'Start': datetime(2025, 6, 9),
        'Duration': 14,
        'Category': 'Monitoring',
        'Phase': 'Phase 6: Launch'
    },
    {
        'Task': 'Documentation & Knowledge Transfer',
        'Start': datetime(2025, 6, 16),
        'Duration': 14,
        'Category': 'Documentation',
        'Phase': 'Phase 6: Launch'
    },
    {
        'Task': 'Citizen Education & Outreach Campaign',
        'Start': datetime(2025, 6, 16),
        'Duration': 14,
        'Category': 'Outreach',
        'Phase': 'Phase 6: Launch'
    },
    {
        'Task': 'Final Thesis Compilation & Submission',
        'Start': datetime(2025, 6, 23),
        'Duration': 8,
        'Category': 'Academic',
        'Phase': 'Phase 6: Launch'
    }
]

# Convert to DataFrame
df = pd.DataFrame(tasks)
df['End'] = df['Start'] + pd.to_timedelta(df['Duration'], unit='D')

# Define colors for different categories
category_colors = {
    'Initiation': '#FF6B6B',
    'Research': '#4ECDC4',
    'Legal': '#45B7D1',
    'Technical': '#96CEB4',
    'Integration': '#FFEAA7',
    'Linguistic': '#DDA0DD',
    'Architecture': '#98D8C8',
    'Data': '#F7DC6F',
    'Security': '#BB8FCE',
    'AI/ML': '#85C1E9',
    'Infrastructure': '#F8C471',
    'Core Development': '#82E0AA',
    'NLP': '#AED6F1',
    'Database': '#F9E79F',
    'Privacy': '#D7BDE2',
    'Frontend': '#A9DFBF',
    'Compliance': '#F5B7B1',
    'Testing': '#D5A6BD',
    'Performance': '#A3E4D7',
    'UAT': '#FAD7A0',
    'Training': '#D2B4DE',
    'Deployment': '#ABEBC6',
    'Monitoring': '#F7DC6F',
    'Documentation': '#D6EAF8',
    'Outreach': '#FADBD8',
    'Academic': '#E8DAEF'
}

# Create the Gantt chart optimized for PNG export
fig, ax = plt.subplots(figsize=(24, 18))  # Larger size for better PNG readability
fig.patch.set_facecolor('white')  # Ensure white background

# Plot bars
for i, task in df.iterrows():
    start_date_plot = task['Start']
    duration = task['Duration']
    category = task['Category']
    
    # Create bar
    bar = ax.barh(i, duration, left=start_date_plot, 
                  color=category_colors.get(category, '#95A5A6'),
                  alpha=0.8, height=0.6)
    
    # Add task name on the bar with larger font for PNG readability
    ax.text(start_date_plot + timedelta(days=duration/2), i, 
            task['Task'][:35] + ('...' if len(task['Task']) > 35 else ''),
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Customize the chart
ax.set_ylim(-0.5, len(df) - 0.5)
ax.set_xlim(start_date - timedelta(days=5), end_date + timedelta(days=5))

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

# Set labels and title with larger fonts for PNG
ax.set_xlabel('Timeline', fontsize=14, fontweight='bold')
ax.set_ylabel('Project Tasks', fontsize=14, fontweight='bold')
ax.set_title('CivicCatalyst AI Toolkit - Comprehensive Project Gantt Chart\n'
             'Master Thesis Project Timeline (September 9, 2024 - July 1, 2025)', 
             fontsize=18, fontweight='bold', pad=25)

# Set y-axis labels with larger font
ax.set_yticks(range(len(df)))
ax.set_yticklabels([f"{i+1:2d}. {task['Task']}" for i, task in df.iterrows()], 
                   fontsize=10)

# Format x-axis with larger font
plt.xticks(rotation=45, fontsize=10)

# Add phase separators and labels
phases = df['Phase'].unique()
phase_positions = []
current_pos = 0

for phase in phases:
    phase_tasks = df[df['Phase'] == phase]
    phase_start = current_pos
    phase_end = current_pos + len(phase_tasks) - 1
    phase_positions.append((phase, phase_start, phase_end))
    current_pos += len(phase_tasks)
    
    # Add horizontal line to separate phases
    if current_pos < len(df):
        ax.axhline(y=current_pos - 0.5, color='red', linestyle='--', alpha=0.7, linewidth=2)

# Add phase labels on the right with larger font
for phase, start, end in phase_positions:
    middle = (start + end) / 2
    ax.text(end_date + timedelta(days=2), middle, phase, 
            rotation=0, va='center', ha='left', fontweight='bold', 
            fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))

# Create legend with larger font
unique_categories = df['Category'].unique()
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=category_colors.get(cat, '#95A5A6'), 
                                alpha=0.8, label=cat) for cat in unique_categories]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1), 
          fontsize=10, title='Task Categories', title_fontsize=12)

# Add grid
ax.grid(True, alpha=0.3)

# Add project milestones
milestones = [
    (datetime(2024, 10, 31), 'Discovery Phase Complete'),
    (datetime(2024, 12, 15), 'Architecture Design Complete'),
    (datetime(2025, 3, 31), 'Core Development Complete'),
    (datetime(2025, 5, 5), 'UI/Integration Complete'),
    (datetime(2025, 6, 16), 'Testing & Validation Complete'),
    (datetime(2025, 7, 1), 'Project Launch & Thesis Submission')
]

for milestone_date, milestone_name in milestones:
    ax.axvline(x=milestone_date, color='red', linestyle='-', alpha=0.8, linewidth=3)
    ax.text(milestone_date, len(df) + 0.5, milestone_name, 
            rotation=90, ha='right', va='bottom', fontweight='bold', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))

# Add project statistics with larger font
total_days = (end_date - start_date).days
stats_text = f"""
Project Statistics:
• Total Duration: {total_days} days (~10 months)
• Total Tasks: {len(df)}
• Development Phases: 6
• Key Deliverables: Privacy-Preserving AI Toolkit
• Target Deployment: REMACTO Platform
• Languages Supported: Arabic, French, Darija
"""

ax.text(start_date, -4, stats_text, fontsize=11, 
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))

# Optimize layout for PNG export
plt.tight_layout()
plt.subplots_adjust(right=0.82, bottom=0.20, top=0.92, left=0.15)

# Save as high-quality PNG file
plt.savefig('CivicCatalyst_Project_Gantt_Chart.png', 
            dpi=300,  # High resolution for crisp text
            bbox_inches='tight',  # Ensures all elements are included
            facecolor='white',  # White background
            edgecolor='none',  # No border
            format='png',
            pad_inches=0.2)  # Small padding around the chart

print("✅ Gantt chart saved as 'CivicCatalyst_Project_Gantt_Chart.png'")
print("📊 High-resolution PNG file created with 300 DPI for clear readability")

# Print task summary
print("\n" + "="*80)
print("🎯 CIVICCATALYST AI TOOLKIT - PROJECT TASK SUMMARY")
print("="*80)
print(f"📅 Project Duration: {start_date.strftime('%B %d, %Y')} - {end_date.strftime('%B %d, %Y')}")
print(f"📋 Total Tasks: {len(df)}")
print(f"⏱️  Total Project Days: {(end_date - start_date).days} days (~10 months)")

print(f"\n📂 TASKS BY PHASE:")
print("-" * 50)
for phase in df['Phase'].unique():
    phase_tasks = df[df['Phase'] == phase]
    phase_duration = phase_tasks['Duration'].sum()
    print(f"\n🔹 {phase} ({len(phase_tasks)} tasks, {phase_duration} total days):")
    for _, task in phase_tasks.iterrows():
        start_str = task['Start'].strftime('%m/%d')
        end_str = (task['Start'] + timedelta(days=task['Duration'])).strftime('%m/%d')
        print(f"   • {task['Task']} ({task['Duration']} days: {start_str}-{end_str})")

print(f"\n🏷️  TASKS BY CATEGORY:")
print("-" * 50)
for category in sorted(df['Category'].unique()):
    cat_tasks = df[df['Category'] == category]
    total_duration = cat_tasks['Duration'].sum()
    print(f"   🔸 {category}: {len(cat_tasks)} tasks, {total_duration} total days")

print(f"\n🎯 KEY MILESTONES:")
print("-" * 50)
milestones = [
    (datetime(2024, 10, 31), 'Discovery Phase Complete'),
    (datetime(2024, 12, 15), 'Architecture Design Complete'),
    (datetime(2025, 3, 31), 'Core Development Complete'),
    (datetime(2025, 5, 5), 'UI/Integration Complete'),
    (datetime(2025, 6, 16), 'Testing & Validation Complete'),
    (datetime(2025, 7, 1), 'Project Launch & Thesis Submission')
]

for milestone_date, milestone_name in milestones:
    print(f"   🎉 {milestone_date.strftime('%B %d, %Y')}: {milestone_name}")

print(f"\n💡 PROJECT HIGHLIGHTS:")
print("-" * 50)
print("   🔐 Privacy-by-design architecture with CNDP compliance")
print("   🌍 Multilingual support (Arabic, French, Darija)")
print("   🤖 Advanced AI/ML with federated learning")
print("   🏛️  Municipal governance transformation")
print("   📱 Mobile-first citizen engagement")
print("   🔒 End-to-end encryption and privacy preservation")

plt.close()  # Close the figure to free memory