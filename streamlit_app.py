
# """
# Multimodal Streamlit App - Main Entry Point
# """
# import streamlit as st
# import sys
# import os

# # Add src directory to path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# from src.ui.main_interface import MainInterface

# def main():
#     """Main application entry point"""
#     try:
#         # Create and run the main interface
#         interface = MainInterface()
#         interface.run()
        
#     except Exception as e:
#         st.error(f"Application error: {str(e)}")
#         st.info("Please check the console for detailed error information.")
        
#         # Show error details in expander for debugging
#         with st.expander("Error Details"):
#             st.exception(e)

# if __name__ == "__main__":
#     main()



"""
Multimodal Streamlit App - Main Entry Point
"""
import streamlit as st
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ui.main_interface import MainInterface





def main():
    """Main application entry point"""
    try:
        st.set_page_config(page_title="Screening", page_icon="📷", layout="wide")

        # Create and run the main interface
        interface = MainInterface()
        interface.run()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()

