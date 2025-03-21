import streamlit as st
import pandas as pd
from datetime import datetime
from auth_system import require_auth, verify_jwt_token

@require_auth
def main():
    """
    This page serves as a personal dashboard for an authenticated citizen/user.
    It includes multiple sections demonstrating how you can enhance citizen 
    participation and engagement with municipal projects, proposals, feedback, 
    and more.
    """
    
    st.title("👤 User View (My Dashboard)")
    st.write("Welcome to your personal dashboard! Engage with ongoing projects, share feedback, track proposals, and more.")

    # Retrieve the current JWT token from session_state
    token = st.session_state.get('jwt_token', None)
    
    if token:
        is_valid, username, role = verify_jwt_token(token)
        if is_valid:
            st.success(f"You are logged in as: **{username}** (Role: **{role}**)")
            
            # --------------------------------------------------------------------------------
            # 1) Profile & User Info
            # --------------------------------------------------------------------------------
            with st.expander("👤 Profile Information", expanded=False):
                """
                In a real-world app, you would query your DB or CSV for user-specific details 
                like full name, municipality, subscription preferences, etc.
                """
                # Example placeholder:
                user_profile = {
                    "Full Name": "John Doe",
                    "City": "Casablanca",
                    "Email": f"{username}@example.com",
                    "Interests": ["Green Spaces", "Education", "Transportation"]
                }
                st.write("#### Basic Profile")
                for key, value in user_profile.items():
                    st.write(f"**{key}:** {value}")
                
                # Example to let the user update some preferences:
                st.write("#### Update Your Notifications Settings")
                email_notifs = st.checkbox("Receive Email Notifications", value=True)
                sms_notifs = st.checkbox("Receive SMS Notifications", value=False)
                if st.button("Save Notification Preferences"):
                    # In production, update the DB or CSV with new settings
                    st.success("Your notification preferences have been saved.")

            # --------------------------------------------------------------------------------
            # 2) Activity Overview & Timeline
            # --------------------------------------------------------------------------------
            with st.expander("🗓️ Activity Timeline", expanded=True):
                """
                Show the user’s recent actions or interactions. In production, you'd
                fetch this data from your logs or DB. Below is just a placeholder.
                """
                timeline_data = [
                    {"Date": "2023-09-01", "Action": "Logged In", "Details": "Successful login from web"},
                    {"Date": "2023-09-02", "Action": "Commented", "Details": "Gave feedback on Project #123"},
                    {"Date": "2023-09-05", "Action": "Proposed Idea", "Details": "Suggested new recycling bins"},
                    {"Date": "2023-09-07", "Action": "Voted", "Details": "Upvoted a local park improvement proposal"},
                ]
                df_timeline = pd.DataFrame(timeline_data)
                st.dataframe(df_timeline)
                
                st.write("Below is a simple bar chart of your actions over time:")
                action_counts = df_timeline["Action"].value_counts()
                st.bar_chart(action_counts)

            # --------------------------------------------------------------------------------
            # 3) Ongoing Municipal Projects (User-Specific or Global)
            # --------------------------------------------------------------------------------
            with st.expander("🏗️ Ongoing Projects in Your Municipality", expanded=False):
                """
                You can display a curated list of projects relevant to the user's city,
                or all active projects. Possibly filter by user interest (like `Green Spaces`).
                """
                projects = [
                    {"Project ID": 101, "Name": "Green Park Expansion", "Status": "In-Progress", "Your Involvement": "Voted + Commented"},
                    {"Project ID": 102, "Name": "New Community Library", "Status": "Planning", "Your Involvement": "No Activity"},
                    {"Project ID": 123, "Name": "Local Sports Field Renovation", "Status": "Ongoing", "Your Involvement": "Commented"},
                ]
                df_projects = pd.DataFrame(projects)
                st.table(df_projects)

                # Show a progress bar for each project (demo purpose)
                for _, row in df_projects.iterrows():
                    st.write(f"**{row['Name']}** (Status: {row['Status']})")
                    # Artificial "progress" example:
                    if row['Status'] == "Planning":
                        st.progress(0.2)
                    elif row['Status'] == "In-Progress" or row['Status'] == "Ongoing":
                        st.progress(0.6)
                    else:
                        st.progress(1.0)

                # Let user subscribe to a project’s updates
                project_subscribe = st.selectbox("Subscribe to a Project for Updates:", [p["Name"] for p in projects])
                if st.button("Subscribe"):
                    # Save subscription in DB
                    st.success(f"You are now subscribed to updates for: {project_subscribe}")

            # --------------------------------------------------------------------------------
            # 4) Citizen Proposals & Voting
            # --------------------------------------------------------------------------------
            with st.expander("🌱 Submit a New Proposal", expanded=False):
                """
                Citizens can propose new ideas or improvements. Typically, you'd store these 
                proposals in your database. People can then upvote/downvote or comment on them.
                """
                st.write("Have an idea to improve your city? Submit it here:")
                proposal_title = st.text_input("Proposal Title", placeholder="e.g., Create More Cycling Lanes")
                proposal_desc = st.text_area("Proposal Description", height=100,
                                             placeholder="Describe your idea, potential benefits, target areas, etc.")
                if st.button("Submit Proposal"):
                    # Save to DB or CSV
                    st.success("Thank you! Your proposal has been submitted. It will be reviewed by the council.")

            with st.expander("🔼 Proposals Voting", expanded=False):
                """
                Show existing proposals and let the user upvote/downvote or comment. 
                This fosters participation and direct feedback from citizens.
                """
                proposals_data = [
                    {"Proposal ID": 1, "Title": "New Recycling Stations", "Upvotes": 35, "Downvotes": 3},
                    {"Proposal ID": 2, "Title": "Monthly Street Market", "Upvotes": 21, "Downvotes": 5},
                    {"Proposal ID": 3, "Title": "Renovate Abandoned Building", "Upvotes": 42, "Downvotes": 10},
                ]
                df_proposals = pd.DataFrame(proposals_data)
                st.dataframe(df_proposals[["Proposal ID", "Title", "Upvotes", "Downvotes"]])

                # Let the user pick a proposal to vote on
                selected_proposal = st.selectbox("Select a Proposal to Vote On:", df_proposals["Title"])
                vote_col1, vote_col2 = st.columns(2)
                with vote_col1:
                    if st.button("Upvote"):
                        st.success(f"You upvoted: {selected_proposal}")
                with vote_col2:
                    if st.button("Downvote"):
                        st.warning(f"You downvoted: {selected_proposal}")

            # --------------------------------------------------------------------------------
            # 5) Personal Sentiment & Topic Tracking
            # --------------------------------------------------------------------------------
            with st.expander("💬 Your Comments Sentiment Summary", expanded=False):
                """
                If you track each user's comments in a DB, you could run a sentiment 
                analysis on them (using your existing pipeline) and show them how 
                they typically comment or engage.
                """
                sample_comments = [
                    {"Date": "2023-09-02", "Comment": "هاذ المشروع زوين بزاف!", "Sentiment": "Positive"},
                    {"Date": "2023-09-03", "Comment": "هاد الفكرة ما عندها حتى معنى!", "Sentiment": "Negative"},
                ]
                df_comments = pd.DataFrame(sample_comments)
                st.write("### Your Recent Comments & Detected Sentiment")
                st.table(df_comments)

                # Quick stats
                pos_count = df_comments[df_comments["Sentiment"] == "Positive"].shape[0]
                neg_count = df_comments[df_comments["Sentiment"] == "Negative"].shape[0]
                total_comments = len(df_comments)
                if total_comments:
                    pos_percentage = round((pos_count / total_comments)*100, 1)
                    neg_percentage = round((neg_count / total_comments)*100, 1)
                    st.write(f"Positive Comments: {pos_count} ({pos_percentage}%)")
                    st.write(f"Negative Comments: {neg_count} ({neg_percentage}%)")
                else:
                    st.write("You haven't commented yet.")

            # --------------------------------------------------------------------------------
            # 6) Feedback to Municipalities or Admins
            # --------------------------------------------------------------------------------
            with st.expander("🗣️ Provide Feedback to Municipal Leaders", expanded=False):
                """
                This feature allows a direct communication channel. 
                Could be a dedicated form or chat that sends data to the municipality.
                """
                st.write("Have questions, concerns, or private feedback about city projects? Let us know.")
                feedback_msg = st.text_area("Enter your feedback", height=100)
                urgency_level = st.select_slider("Urgency Level", ["Low", "Medium", "High"])
                if st.button("Send Feedback"):
                    # In production, you'd store or email this to the city
                    st.success("Your feedback has been sent to the municipal team.")
            
            # --------------------------------------------------------------------------------
            # 7) Advanced Tools (e.g., ChatGPT-based Analysis, if desired)
            # --------------------------------------------------------------------------------
            with st.expander("🤖 AI Insights & Summaries", expanded=False):
                """
                Optionally, you might let the user ask for an AI-generated summary of 
                proposals or comments. This is where you could integrate your GPT 
                or other LLM logic at a user level.
                """
                st.write("Ask the AI for a summary of top concerns or ideas in your municipality:")
                user_query = st.text_area("Your question to the AI", placeholder="E.g., What are people saying about public parks?")
                if st.button("Get AI Summary"):
                    st.info("Calling LLM with your question...")
                    # In production, you would call your OpenAI / GPT function with user_query
                    # and relevant context from the user’s city or data. This is just a placeholder:
                    fake_response = "Most citizens appreciate green areas, but worry about ongoing maintenance costs."
                    st.success(f"AI Summary:\n{fake_response}")

            # --------------------------------------------------------------------------------
            # 8) Documents & Data Access
            # --------------------------------------------------------------------------------
            with st.expander("📂 Shared Documents & Data", expanded=False):
                """
                Display or allow download of relevant documents or data pertaining to 
                city projects. This fosters transparency and open data access.
                """
                st.write("Below are public resources or city planning documents you may find useful:")
                sample_docs = {
                    "2023 Budget Overview": "budget_2023.pdf",
                    "Community Survey Results": "survey_results.csv",
                }
                for doc_title, doc_file in sample_docs.items():
                    st.write(f"**{doc_title}**")
                    # Provide a download button
                    st.download_button(
                        label=f"Download {doc_title}",
                        data="Some file bytes here",  # In a real scenario, you'd open/read the file
                        file_name=doc_file
                    )

            # --------------------------------------------------------------------------------
            # 9) Achievements / Badges (Gamification)
            # --------------------------------------------------------------------------------
            with st.expander("🏆 Your Civic Achievements", expanded=False):
                """
                Simple gamification: awarding badges for participation (e.g., 'First Comment', 
                'Frequent Voter', 'Top Contributor'). This encourages more engagement.
                """
                achievements = [
                    {"Badge": "First Comment", "Earned On": "2023-09-02"},
                    {"Badge": "Frequent Voter", "Earned On": "2023-09-07"},
                ]
                st.table(pd.DataFrame(achievements))

                st.write("Keep engaging with proposals and comments to earn new badges!")

        else:
            st.warning("Token is invalid or expired. Please log in again.")
    else:
        st.info("No token found in session. Please go back and log in.")

if __name__ == "__main__":
    main()
