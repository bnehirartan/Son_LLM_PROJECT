import gradio as gr
import database as db
from financial_assistant import generate_financial_response 
from riskanalyzer import get_response
import json
import pandas as pd
import uuid

# Global user session state
current_user = None
def generate_risk_excel(risk_factors):
    # risk_factors: list of dicts
    df = pd.DataFrame(risk_factors)
    filename = f"risk_factors_{uuid.uuid4().hex}.xlsx"
    path = f"C:\\Users\\user\\Desktop\\deneme_sınllm\\{filename}"
    df.to_excel(path, index=False)
    return path

def login(username, password):
    """Login function for the interface"""
    global current_user
    success, result = db.authenticate_user(username, password)
    
    if success:
        current_user = result
        # Risk analyzer modülündeki current_user_id'yi güncelle
        import riskanalyzer
        riskanalyzer.current_user_id = current_user
        
        # Try to load user profile
        profile_success, profile_data = db.get_user_profile(current_user)
        if profile_success:
            # User has an existing profile, we'll need to populate form in the load_profile function
            return "Login successful!", gr.update(visible=False), gr.update(visible=True)
        else:
            # User doesn't have a profile yet
            return "Login successful! Please complete your profile.", gr.update(visible=False), gr.update(visible=True)
    else:
        return f"Login failed: {result}", gr.update(visible=True), gr.update(visible=False)

def register(username, password, email, confirm_password):
    """Register function for the interface"""
    if password != confirm_password:
        return "Passwords do not match", gr.update(visible=True), gr.update(visible=False)
    
    success, message = db.register_user(username, password, email)
    
    if success:
        return f"{message}. Please login.", gr.update(visible=True), gr.update(visible=False)
    else:
        return message, gr.update(visible=True), gr.update(visible=False)

def logout():
    """Logout function for the interface"""
    global current_user
    current_user = None
    # Risk analyzer modülündeki current_user_id'yi sıfırla
    import riskanalyzer
    riskanalyzer.current_user_id = None
    return gr.update(visible=True), gr.update(visible=False)

def save_profile(risk_taker, risk_word, game_show, investment_allocation, 
                market_follow, new_investment, buy_things, finance_reading,
                previous_investments, investment_goal):
    """Save user profile information to the database"""
    if current_user is None:
        return "Error: User not logged in"
    
    # Checkbox groups için özel işlem, liste şeklinde geliyorlar
    previous_investments_list = previous_investments if isinstance(previous_investments, list) else []
    
    success, message = db.save_user_profile(
        current_user,
        risk_taker, risk_word, game_show, investment_allocation,
        market_follow, new_investment, buy_things, finance_reading,
        previous_investments_list, investment_goal
    )
    
    if success:
        return "Profile saved successfully!"
    else:
        return f"Error saving profile: {message}"

def load_profile():
    """Load user profile from database"""
    if current_user is None:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), 
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    
    success, profile_data = db.get_user_profile(current_user)
    
    if not success:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), 
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    
    # Return updates for all fields
    return (
        gr.update(value=profile_data.get('risk_taker', 'Cautious')),
        gr.update(value=profile_data.get('risk_word', 'Uncertainty')),
        gr.update(value=profile_data.get('game_show', '$1,000 in cash')),
        gr.update(value=profile_data.get('investment_allocation', '60% in low-risk, 30% in medium-risk, 10% in high-risk investments')),
        gr.update(value=profile_data.get('market_follow', 'Weekly')),
        gr.update(value=profile_data.get('new_investment', 'Research thoroughly before investing')),
        gr.update(value=profile_data.get('buy_things', 'Neutral')),
        gr.update(value=profile_data.get('finance_reading', 'Neutral')),
        gr.update(value=profile_data.get('previous_investments', [])),
        gr.update(value=profile_data.get('investment_goal', 'Long-term savings'))
    )

# Financial Education chatbot fonksiyonu - generate_financial_response'u kullanarak
def education_chatbot(message, pdf_file, history):
    """Financial education chatbot using the financial_assistant module"""
    try:
        # PDF dosyası yüklenmişse, debug bilgisi görüntüle
        
        
        # Chat history'yi generate_financial_response fonksiyonuna geçir
        response = generate_financial_response(message, pdf_file, history)
        
        # HTML içeriğini Gradio chatbot için düzenle
        processed_response = response
        
        # Eğer içerik HTML içeriyorsa, bazı tagları temizle
        if "<div" in response or "<a" in response:
            # Link yapısını daha basit hale getir ama fonksiyonelliği koru
            processed_response = response.replace('<div style=\'font-family: sans-serif; line-height: 1.6\'>', '')
            processed_response = processed_response.replace('</div>', '')
            processed_response = processed_response.replace('<br><br><b>Kaynaklar:</b><br>', '\n\nKaynaklar:\n')
            processed_response = processed_response.replace('<br>', '\n')
            
            # Linkleri düzenle
            import re
            links = re.findall(r'<a href="([^"]+)" target="_blank">([^<]+)</a>', processed_response)
            for link_url, link_text in links:
                processed_response = processed_response.replace(f'<a href="{link_url}" target="_blank">{link_text}</a>', link_url)
        
        return history + [[message, processed_response]]
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Education chatbot hata: {e}\n{error_details}")
        return history + [[message, f"Bir hata oluştu: {str(e)}"]]

def advisor_chatbot(message, history):
    """Risk & Analyzer chatbot using the riskanalyzer module"""
    try:
        response = get_response(message)
        excel_file = None
        
        # Eğer response bir dictionary ise, formatla
        if isinstance(response, dict):
            if "risk_factors_table" in response:
                formatted_response = "📊 Risk Analysis Results:\n\n"
                formatted_response += "Risk Factors Table:\n"
                for row in response["risk_factors_table"]:
                    formatted_response += f"• {row.get('Factor', row.get('factor'))}: " \
                                        f"{row.get('Status', row.get('value'))} – " \
                                        f"{row.get('Comment', row.get('description'))}\n"

                raw_summary = response.get("summary", "No summary provided.")
                import re, json
                cleaned = re.sub(r'```json|```', '', raw_summary, flags=re.IGNORECASE).strip()
                cleaned = re.sub(r'^\s*json', '', cleaned, flags=re.IGNORECASE).strip()

                try:
                    parsed = json.loads(cleaned)
                    summary_text = parsed.get("summary", "")
                except (json.JSONDecodeError, TypeError):
                    summary_text = raw_summary

                formatted_response += f"\nSummary:\n{summary_text}"
                
                # Excel dosyasını oluştur
                excel_path = generate_risk_excel(response["risk_factors_table"])
                
                # gr.File bileşenini görünür yapmak için None olmayan bir değer döndür
                return history + [[message, formatted_response]], gr.update(value=excel_path, visible=True)

            elif response.get("status") == "success" and "result" in response:
                formatted_response = f"```\n{response['result']}\n```"
            elif "<div" in response or "<a" in response:
                formatted_response = response.replace('<div style=\'font-family: sans-serif; line-height: 1.6\'>', '')
                formatted_response = formatted_response.replace('</div>', '')
                formatted_response = formatted_response.replace('<br><br><b>Kaynaklar:</b><br>', '\n\nKaynaklar:\n')
                formatted_response = formatted_response.replace('<br>', '\n')
                
                import re
                links = re.findall(r'<a href="([^"]+)" target="_blank">([^<]+)</a>', formatted_response)
                for link_url, link_text in links:
                    formatted_response = formatted_response.replace(f'<a href="{link_url}" target="_blank">{link_text}</a>', link_url)
        else:
            formatted_response = str(response)
            
        return history + [[message, formatted_response]], gr.update(value=None, visible=False)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Risk analyzer chatbot error: {e}\n{error_details}")
        return history + [[message, f"An error occurred: {str(e)}"]], gr.update(value=None, visible=False)

# Create custom CSS for larger text with black and white theme
custom_css = """
:root {
    --primary-color: #000000;
    --secondary-color: #333333;
    --light-gray: #CCCCCC;
    --dark-gray: #222222;
    --text-color: #000000;
    --light-text: #FFFFFF;
    --accent-color: #555555;
    --border-radius: 8px;
    --box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
}

.larger-text label {
    font-size: 22px !important;
    font-weight: bold !important;
    color: var(--primary-color) !important;
}

.larger-text .prose h1 {
    font-size: 36px !important;
    font-weight: bold !important;
    color: var(--primary-color) !important;
}

.larger-text .prose h2 {
    font-size: 32px !important;
    font-weight: bold !important;
    color: var(--primary-color) !important;
}

.larger-text .prose p {
    font-size: 20px !important;
    color: var(--text-color) !important;
}

.larger-text input, .larger-text select {
    font-size: 20px !important;
    padding: 12px !important;
    border-color: var(--secondary-color) !important;
}

.larger-text button {
    font-size: 20px !important;
    padding: 10px 20px !important;
    font-weight: bold !important;
    background-color: var(--primary-color) !important;
    color: var(--light-text) !important;
    border-radius: var(--border-radius) !important;
    transition: all 0.2s ease !important;
    margin: 5px !important;
}

.larger-text button:hover {
    background-color: var(--secondary-color) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--box-shadow) !important;
}

/* Override Gradio defaults for better visibility */
.dark .larger-text label, .dark .larger-text .prose h1, .dark .larger-text .prose h2 {
    color: var(--light-text) !important;
}

.dark .larger-text .prose p {
    color: #E0E0E0 !important;
}

.dark .larger-text input, .dark .larger-text select {
    background-color: var(--dark-gray) !important;
    color: white !important;
    border-color: var(--light-gray) !important;
}

/* Make dropdown options more readable */
.larger-text select option {
    font-size: 20px !important;
    padding: 10px !important;
}

/* Chatbox özel stilleri */
.large-chatbox {
    font-size: 18px !important;
    border-radius: 12px !important;
    box-shadow: var(--box-shadow) !important;
    margin-bottom: 20px !important;
    border: 1px solid var(--light-gray) !important;
}

.large-chatbox .message {
    padding: 15px !important;
    border-radius: 10px !important;
    margin: 8px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
}

.large-chatbox .user-message {
    background-color: var(--secondary-color) !important;
    color: var(--light-text) !important;
    font-size: 18px !important;
    font-weight: 500 !important;
    margin-left: 20px !important;
    margin-right: 8px !important;
}

.large-chatbox .bot-message {
    background-color: var(--light-gray) !important;
    color: var(--text-color) !important;
    font-size: 18px !important;
    margin-right: 20px !important;
    margin-left: 8px !important;
}

/* Chatbox dark mode */
.dark .large-chatbox {
    border-color: #444 !important;
}

.dark .large-chatbox .bot-message {
    background-color: var(--dark-gray) !important;
    color: var(--light-text) !important;
}

/* Belirgin input alanı */
.prominent-input {
    border: 2px solid var(--accent-color) !important;
    border-radius: var(--border-radius) !important;
    background-color: rgba(255, 255, 255, 0.05) !important;
    transition: all 0.3s ease !important;
    font-size: 18px !important;
    padding: 12px !important;
    box-shadow: var(--box-shadow) !important;
    margin-bottom: 10px !important;
}

.prominent-input:focus-within {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.2) !important;
}

.dark .prominent-input {
    background-color: rgba(0, 0, 0, 0.2) !important;
}

.dark .prominent-input:focus-within {
    box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.2) !important;
}

/* PDF yükleme alanı */
.pdf-upload-area {
    border: 2px dashed var(--accent-color) !important;
    border-radius: var(--border-radius) !important;
    padding: 15px !important;
    background-color: rgba(0, 0, 0, 0.02) !important;
    transition: all 0.3s ease !important;
    text-align: center !important;
    height: 100% !important;
    min-height: 150px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    box-shadow: var(--box-shadow) !important;
}

.pdf-upload-area:hover {
    border-color: var(--primary-color) !important;
    background-color: rgba(0, 0, 0, 0.05) !important;
}

.dark .pdf-upload-area {
    background-color: rgba(255, 255, 255, 0.05) !important;
}

.dark .pdf-upload-area:hover {
    background-color: rgba(255, 255, 255, 0.1) !important;
}

/* Aksiyon butonları */
.action-button {
    background-color: var(--primary-color) !important;
    min-width: 120px !important;
}

.action-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
}
"""

# Create the main application
with gr.Blocks(title="FINSENTIO", css=custom_css, theme=gr.themes.Monochrome()) as app:
    with gr.Blocks(elem_classes=["larger-text"]):
        gr.Markdown("# FINSENTIO")
        
        # Create dashboard interface
        with gr.Group(visible=False) as dashboard:
            gr.Markdown("# Welcome to your Dashboard")
            
            with gr.Tabs():
                with gr.TabItem("Profile"):
                    gr.Markdown("## User Risk Profile Assessment")
                    gr.Markdown("Please answer the following questions to help us understand your investment preferences and risk tolerance.")
                    
                    # Question 1
                    with gr.Row():
                        with gr.Column():
                            risk_taker = gr.Radio(
                                label="1. How would your best friend describe you as a risk taker?",
                                choices=[
                                    "A real gambler",
                                    "Willing to take risks after completing adequate research",
                                    "Cautious",
                                    "A real risk avoider"
                                ],
                                value="Cautious",
                                scale=2
                            )
                    
                    # Question 2
                    with gr.Row():
                        with gr.Column():
                            risk_word = gr.Radio(
                                label="2. When you think of the word \"risk\", which of the following words comes to mind first?",
                                choices=[
                                    "Loss",
                                    "Uncertainty",
                                    "Opportunity",
                                    "Thrill"
                                ],
                                value="Uncertainty",
                                scale=2
                            )
                    
                    # Question 3
                    with gr.Row():
                        with gr.Column():
                            game_show = gr.Radio(
                                label="3. You are on a TV game show and can choose one of the following. Which would you take?",
                                choices=[
                                    "$1,000 in cash",
                                    "A 50% chance at winning $5,000",
                                    "A 25% chance at winning $10,000",
                                    "A 5% chance at winning $100,000"
                                ],
                                value="$1,000 in cash",
                                scale=2
                            )
                    
                    # Question 4
                    with gr.Row():
                        with gr.Column():
                            investment_allocation = gr.Radio(
                                label="4. If you had to invest $20,000, which of the following allocations would you find most appealing?",
                                choices=[
                                    "60% in low-risk, 30% in medium-risk, 10% in high-risk investments",
                                    "30% in low-risk, 30% in medium-risk, 40% in high-risk investments",
                                    "10% in low-risk, 40% in medium-risk, 50% in high-risk investments"
                                ],
                                value="60% in low-risk, 30% in medium-risk, 10% in high-risk investments",
                                scale=2
                            )
                    
                    # Question 5
                    with gr.Row():
                        with gr.Column():
                            market_follow = gr.Radio(
                                label="5. How often do you follow financial markets?",
                                choices=[
                                    "Daily",
                                    "Weekly",
                                    "Occasionally",
                                    "Never"
                                ],
                                value="Weekly",
                                scale=2
                            )
                    
                    # Question 6
                    with gr.Row():
                        with gr.Column():
                            new_investment = gr.Radio(
                                label="6. What would you do when you hear about a new investment opportunity?",
                                choices=[
                                    "Immediately jump in",
                                    "Research thoroughly before investing",
                                    "Ask others first, then decide",
                                    "Wait and observe over time"
                                ],
                                value="Research thoroughly before investing",
                                scale=2
                            )
                    
                    # Question 7
                    with gr.Row():
                        with gr.Column():
                            buy_things = gr.Radio(
                                label="7. I rarely buy things I don't need.",
                                choices=[
                                    "Agree",
                                    "Neutral",
                                    "Disagree"
                                ],
                                value="Neutral",
                                scale=2
                            )
                    
                    # Question 8
                    with gr.Row():
                        with gr.Column():
                            finance_reading = gr.Radio(
                                label="8. I like to read about finance and the economy.",
                                choices=[
                                    "Agree",
                                    "Neutral",
                                    "Disagree"
                                ],
                                value="Neutral",
                                scale=2
                            )
                    
                    # Question 9
                    with gr.Row():
                        with gr.Column():
                            previous_investments = gr.CheckboxGroup(
                                label="9. Which of the following have you invested in before? (You can choose more than one)",
                                choices=[
                                    "Stocks",
                                    "Cryptocurrency",
                                    "Foreign currencies",
                                    "Gold or other commodities",
                                    "Fixed deposit accounts",
                                    "I have never invested"
                                ],
                                value=[],
                                scale=2
                            )
                    
                    # Question 10
                    with gr.Row():
                        with gr.Column():
                            investment_goal = gr.Radio(
                                label="10. What is your main goal for investing?",
                                choices=[
                                    "Short-term profit",
                                    "Long-term savings",
                                    "Retirement planning",
                                    "Wealth preservation"
                                ],
                                value="Long-term savings",
                                scale=2
                            )
                    
                    # Save button
                    with gr.Row():
                        save_button = gr.Button("Save Profile", size="lg", variant="primary")
                    profile_message = gr.Markdown("")
                    
                    # Connect save button to function
                    save_button.click(
                        fn=save_profile,
                        inputs=[
                            risk_taker, risk_word, game_show, investment_allocation,
                            market_follow, new_investment, buy_things, finance_reading,
                            previous_investments, investment_goal
                        ],
                        outputs=profile_message
                    )
                
                with gr.TabItem("Education Chatbot"):
                    gr.Markdown("## Financial Education Chatbot")
                    gr.Markdown("Ask any questions about financial terms, concepts, or strategies to improve your knowledge!")
                    
                    # Education chatbot interface - daha büyük chat alanı
                    education_chat = gr.Chatbot(
                        label="Chat History",
                        height=700,  # Daha da yüksek chat alanı
                        value=[],
                        elem_classes=["large-chatbox"],  # Özel CSS sınıfı eklendi
                        show_label=True,  # Etiketi göster
                        container=True  # Container kullan
                    )
                    
                    # Kontrol paneli - alt kısım
                    with gr.Row(equal_height=True):
                        # Sol taraf - Soru girme alanı
                        with gr.Column(scale=4):
                            education_msg = gr.Textbox(
                                label="Your Question",
                                placeholder="Ask me anything about finance...",
                                container=True,
                                lines=2,
                                max_lines=4,
                                elem_classes=["prominent-input"]
                            )
                            
                            # Butonlar alt satırda
                            with gr.Row():
                                education_submit = gr.Button("Ask", variant="primary", size="lg", elem_classes=["action-button"])
                                education_clear = gr.Button("Clear Chat", size="lg")
                        
                        # Sağ taraf - PDF yükleme
                        with gr.Column(scale=1, min_width=200):
                            pdf_upload = gr.File(
                                label="Upload PDF",
                                file_types=[".pdf"],
                                type="binary",
                                elem_classes=["pdf-upload-area"]
                            )
                    
                    # Connect chatbot components
                    education_submit.click(
                        fn=education_chatbot,
                        inputs=[education_msg, pdf_upload, education_chat],
                        outputs=[education_chat],
                        api_name="education_chat"
                    )
                    education_clear.click(lambda: [], None, education_chat)
                
                with gr.TabItem("Financial Adviser & Analyzer Chatbot"):
                    gr.Markdown("## Financial Adviser & Analyzer Chatbot")
                    gr.Markdown("Get personalized financial advice and analysis based on your profile and market conditions.")
                    
                    # Adviser chatbot interface
                    adviser_chat = gr.Chatbot(
                        label="Chat History",
                        height=700,
                        value=[],
                        allow_html=True
                    )
                    adviser_msg = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask for financial advice or market analysis...",
                        scale=7
                    )
                    with gr.Row():
                        adviser_submit = gr.Button("Ask", scale=1, variant="primary")
                        adviser_clear = gr.Button("Clear Chat", scale=1)
                    
                    # Excel download link - interactive=True ekledik ve type="file" belirttik
                    excel_link = gr.File(
                        label="Download Excel Report",
                        visible=False,
                        interactive=True,
                        type="file"
                    )
                    
                    # Connect chatbot components
                    outputs = adviser_submit.click(
                        fn=advisor_chatbot,
                        inputs=[adviser_msg, adviser_chat],
                        outputs=[adviser_chat, excel_link],
                        # api_name="adviser_chat"
                    )
                    adviser_clear.click(lambda: [[], None], None, [adviser_chat, excel_link])
            
            # Logout button
            with gr.Row():
                logout_button = gr.Button("Logout", size="lg", variant="stop")
        
        # Create auth interface
        with gr.Group(visible=True) as auth_interface:
            with gr.Tabs():
                with gr.TabItem("Login"):
                    username_login = gr.Textbox(label="Username", scale=2)
                    password_login = gr.Textbox(label="Password", type="password", scale=2)
                    with gr.Row():
                        login_button = gr.Button("Login", size="lg", variant="primary")
                    login_message = gr.Markdown("")
                
                with gr.TabItem("Register"):
                    username_register = gr.Textbox(label="Username", scale=2)
                    email_register = gr.Textbox(label="Email", scale=2)
                    password_register = gr.Textbox(label="Password", type="password", scale=2)
                    confirm_password = gr.Textbox(label="Confirm Password", type="password", scale=2)
                    with gr.Row():
                        register_button = gr.Button("Register", size="lg", variant="primary")
                    register_message = gr.Markdown("")
        
        # Connect the buttons to functions
        login_button.click(
            fn=login,
            inputs=[username_login, password_login],
            outputs=[login_message, auth_interface, dashboard]
        ).then(
            fn=load_profile,
            inputs=None,
            outputs=[
                risk_taker, risk_word, game_show, investment_allocation,
                market_follow, new_investment, buy_things, finance_reading,
                previous_investments, investment_goal
            ]
        )
        
        register_button.click(
            fn=register,
            inputs=[username_register, password_register, email_register, confirm_password],
            outputs=[register_message, auth_interface, dashboard]
        )
        
        logout_button.click(
            fn=logout,
            outputs=[auth_interface, dashboard]
        )

if __name__ == "__main__":
    app.launch()