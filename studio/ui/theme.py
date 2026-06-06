import gradio as gr

def get_custom_theme():
    # Standard Violet/Indigo theme base
    theme = gr.themes.Default(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="zinc",
        font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
    ).set(
        body_background_fill="*neutral_950",
        body_background_fill_dark="*neutral_950",
        block_background_fill="*neutral_900",
        block_background_fill_dark="*neutral_900",
        block_border_color="*neutral_800",
        block_border_color_dark="*neutral_800",
        button_primary_background_fill="linear-gradient(90deg, #6366f1 0%, #a855f7 100%)",
        button_primary_background_fill_hover="linear-gradient(90deg, #4f46e5 0%, #9333ea 100%)",
        button_primary_text_color="#ffffff",
    )
    
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    body, .gradio-container {
        font-family: 'Outfit', sans-serif !important;
        background-color: #030712 !important;
        background-image: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.15) 0%, rgba(3, 7, 18, 0) 50%) !important;
        background-attachment: fixed !important;
    }
    
    /* Studio Header Typography */
    .studio-title {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #a5b4fc 0%, #e9d5ff 50%, #f472b6 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center;
        margin-bottom: 0.2rem !important;
        letter-spacing: -0.025em;
        text-shadow: 0 4px 20px rgba(168, 85, 247, 0.15);
    }
    
    .studio-subtitle {
        font-size: 1.1rem !important;
        color: #9ca3af !important;
        text-align: center;
        margin-bottom: 2rem !important;
        font-weight: 300;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    /* Glassmorphism Panel styles */
    .gr-panel, .gr-box, .gr-form {
        background: rgba(17, 24, 39, 0.7) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .gr-panel:hover {
        border-color: rgba(99, 102, 241, 0.25) !important;
        box-shadow: 0 8px 32px 0 rgba(99, 102, 241, 0.1) !important;
    }
    
    /* Interactive Cards */
    .stat-card {
        padding: 1.5rem !important;
        text-align: center !important;
        border-radius: 12px !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        transition: transform 0.2s, border-color 0.2s;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(168, 85, 247, 0.3) !important;
    }
    
    /* Buttons */
    .gr-button-primary {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    
    .gr-button-primary:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px 0 rgba(168, 85, 247, 0.4) !important;
    }
    
    .gr-button-secondary {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #e5e7eb !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    .gr-button-secondary:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #030712;
    }
    ::-webkit-scrollbar-thumb {
        background: #1f2937;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #374151;
    }
    """
    return theme, custom_css
