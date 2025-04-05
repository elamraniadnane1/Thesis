import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import streamlit as st
from st_pages import add_page_title, get_nav_from_toml



st.set_page_config(
    page_title="Civic Catalyst",
    page_icon="icon.png",  # Must be a small icon-sized image (like 32x32)
    layout="wide"
)


# If you want to use the no-sections version, this
# defaults to looking in .streamlit/pages.toml, so you can
# just call `get_nav_from_toml()`
nav = get_nav_from_toml(".streamlit/pages_sections.toml")
st.logo("icon.png")
st.image("icon.png", width=200)
pg = st.navigation(nav)
add_page_title(pg)

pg.run()
def main():
    st.title("Civic Catalyst")

    # Check for a stored JWT token in session state
    if "jwt_token" not in st.session_state or not st.session_state["jwt_token"]:
        # If no token, the user is not logged in
        st.write("Redirecting to Login...")
        st.switch_page("pages\\Login.py")  # Name of your Login page
    else:
        # User is logged in; show main content here
        st.success(f"Welcome back, {st.session_state.get('username', 'User')}!")
        st.write("This is the main content of the Civic Catalyst application.")

if __name__ == "__main__":
    main()
