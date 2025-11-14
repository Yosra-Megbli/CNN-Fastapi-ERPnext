"""
ERPNEXT SETUP - Création automatique du Doctype AI_Document
À lancer UNE SEULE FOIS pour configurer ERPNext
"""

import requests
import json
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================
ERPNEXT_URL = "http://localhost:8080"
USERNAME = "Administrator"
PASSWORD = "admin"

# ============================================================================
# DOCTYPE DEFINITION
# ============================================================================
AI_DOCUMENT_DOCTYPE = {
    "doctype": "DocType",
    "name": "AI_Document",
    "module": "Custom",
    "custom": 1,
    "istable": 0,
    "issingle": 0,
    "is_submittable": 0,
    "autoname": "field:filename",
    "naming_rule": "By fieldname",
    "title_field": "filename",
    "fields": [
        {
            "fieldname": "document_info_section",
            "fieldtype": "Section Break",
            "label": "Document Information"
        },
        {
            "fieldname": "filename",
            "fieldtype": "Data",
            "label": "Filename",
            "reqd": 1,
            "unique": 1,
            "in_list_view": 1
        },
        {
            "fieldname": "document_class",
            "fieldtype": "Select",
            "label": "Document Class",
            "options": "Drawing\nInvoice\nReport\nReceipt",
            "reqd": 1,
            "in_list_view": 1,
            "in_standard_filter": 1
        },
        {
            "fieldname": "file_hash",
            "fieldtype": "Data",
            "label": "File Hash (SHA-256)",
            "unique": 1,
            "read_only": 1
        },
        {
            "fieldname": "column_break_1",
            "fieldtype": "Column Break"
        },
        {
            "fieldname": "confidence_score",
            "fieldtype": "Float",
            "label": "Confidence Score",
            "precision": 4,
            "in_list_view": 1
        },
        {
            "fieldname": "uploaded_by",
            "fieldtype": "Link",
            "label": "Uploaded By",
            "options": "User",
            "default": "Administrator"
        },
        {
            "fieldname": "upload_date",
            "fieldtype": "Datetime",
            "label": "Upload Date",
            "default": "Now"
        },
        {
            "fieldname": "ai_analysis_section",
            "fieldtype": "Section Break",
            "label": "AI Analysis"
        },
        {
            "fieldname": "keywords",
            "fieldtype": "Small Text",
            "label": "Keywords"
        },
        {
            "fieldname": "summary",
            "fieldtype": "Long Text",
            "label": "Summary"
        },
        {
            "fieldname": "ocr_text",
            "fieldtype": "Long Text",
            "label": "OCR Extracted Text"
        },
        {
            "fieldname": "security_section",
            "fieldtype": "Section Break",
            "label": "Security"
        },
        {
            "fieldname": "is_encrypted",
            "fieldtype": "Check",
            "label": "Is Encrypted",
            "default": 0
        }
    ],
    "permissions": [
        {
            "role": "System Manager",
            "read": 1,
            "write": 1,
            "create": 1,
            "delete": 1,
            "submit": 0,
            "cancel": 0,
            "amend": 0
        },
        {
            "role": "All",
            "read": 1,
            "write": 0,
            "create": 0,
            "delete": 0
        }
    ]
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def login():
    """Login to ERPNext and get session"""
    print("🔐 Logging in to ERPNext...")
    
    session = requests.Session()
    
    # Login
    login_data = {
        'cmd': 'login',
        'usr': USERNAME,
        'pwd': PASSWORD
    }
    
    response = session.post(
        f"{ERPNEXT_URL}/api/method/login",
        data=login_data
    )
    
    if response.status_code == 200:
        print(f"✅ Logged in as {USERNAME}")
        return session
    else:
        print(f"❌ Login failed: {response.status_code}")
        print(response.text)
        sys.exit(1)

def create_doctype(session):
    """Create AI_Document DocType"""
    print("\n📝 Creating AI_Document DocType...")
    
    # Check if already exists
    check_response = session.get(
        f"{ERPNEXT_URL}/api/resource/DocType/AI_Document"
    )
    
    if check_response.status_code == 200:
        print("⚠️  AI_Document DocType already exists!")
        choice = input("Do you want to recreate it? (yes/no): ")
        
        if choice.lower() == 'yes':
            print("🗑️  Deleting existing DocType...")
            delete_response = session.delete(
                f"{ERPNEXT_URL}/api/resource/DocType/AI_Document"
            )
            if delete_response.status_code not in [200, 202]:
                print("❌ Failed to delete existing DocType")
                sys.exit(1)
        else:
            print("✅ Keeping existing DocType")
            return True
    
    # Create new DocType
    response = session.post(
        f"{ERPNEXT_URL}/api/resource/DocType",
        json=AI_DOCUMENT_DOCTYPE,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code in [200, 201]:
        print("✅ AI_Document DocType created successfully!")
        return True
    else:
        print(f"❌ Failed to create DocType: {response.status_code}")
        print(response.text)
        return False

def create_api_credentials(session):
    """Generate API Key and Secret"""
    print("\n🔑 Generating API credentials...")
    
    # Get user
    user_response = session.get(
        f"{ERPNEXT_URL}/api/resource/User/{USERNAME}"
    )
    
    if user_response.status_code != 200:
        print("❌ Failed to get user")
        return None
    
    # Generate API secret
    generate_response = session.post(
        f"{ERPNEXT_URL}/api/method/frappe.core.doctype.user.user.generate_keys",
        json={'user': USERNAME}
    )
    
    if generate_response.status_code == 200:
        result = generate_response.json()
        api_key = result.get('message', {}).get('api_key')
        api_secret = result.get('message', {}).get('api_secret')
        
        print("✅ API Credentials generated!")
        print(f"\n📋 SAVE THESE CREDENTIALS:\n")
        print(f"API_KEY: {api_key}")
        print(f"API_SECRET: {api_secret}")
        print("\n⚠️  Store these in your .env file!")
        
        return {'api_key': api_key, 'api_secret': api_secret}
    else:
        print("❌ Failed to generate API credentials")
        return None

def test_doctype(session):
    """Test by creating a sample document"""
    print("\n🧪 Testing AI_Document creation...")
    
    test_doc = {
        "doctype": "AI_Document",
        "filename": "test_document.pdf",
        "document_class": "Invoice",
        "file_hash": "test_hash_12345",
        "confidence_score": 0.95,
        "keywords": "test, invoice, sample",
        "summary": "This is a test document",
        "uploaded_by": "Administrator"
    }
    
    response = session.post(
        f"{ERPNEXT_URL}/api/resource/AI_Document",
        json=test_doc,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code in [200, 201]:
        doc_name = response.json().get('data', {}).get('name')
        print(f"✅ Test document created: {doc_name}")
        
        # Delete test document
        session.delete(f"{ERPNEXT_URL}/api/resource/AI_Document/{doc_name}")
        print("🗑️  Test document deleted")
        return True
    else:
        print(f"❌ Test failed: {response.status_code}")
        print(response.text)
        return False

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🚀 ARKEYEZ - ERPNext Setup Script")
    print("="*70)
    
    # Step 1: Login
    session = login()
    
    # Step 2: Create DocType
    if not create_doctype(session):
        print("\n❌ Setup failed at DocType creation")
        sys.exit(1)
    
    # Step 3: Generate API credentials
    credentials = create_api_credentials(session)
    
    # Step 4: Test
    if test_doctype(session):
        print("\n" + "="*70)
        print("✅ SETUP COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        if credentials:
            print("\n📝 Next steps:")
            print("1. Copy API credentials to .env file")
            print("2. Update your main.py to use ERPNextConnector")
            print("3. Restart your FastAPI server")
        
        print("\n🎉 You can now use AI_Document in ERPNext!")
    else:
        print("\n⚠️  Setup completed but test failed")

if __name__ == "__main__":
    main()