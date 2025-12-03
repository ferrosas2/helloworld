
import json
from datetime import datetime

def lambda_handler(event, context):
    """
    Graceful degradation handler that returns predefined, regulation-safe responses.

    This is the final safety net when all models fail, ensuring customers
    always receive helpful guidance that complies with financial regulations.
    """
    prompt = event.get('prompt', '')
    use_case = event.get('use_case', 'general')
    reason = event.get('reason', 'unknown')

    # Regulation-compliant responses by use case
    responses = {
        "general": {
            "response": "I apologize, but I'm currently experiencing technical difficulties and cannot process your request at this time. For immediate assistance, please contact our customer service team at 1-800-555-1234 (available 24/7) or visit your nearest branch location.",
            "contact_info": {
                "phone": "1-800-555-1234",
                "hours": "24/7",
                "alternative": "Visit nearest branch"
            }
        },
        "product_question": {
            "response": "I'm unable to access our product information systems at the moment. For detailed information about our financial products and services, please:\n\n1. Call our product specialists at 1-800-555-PROD (1-800-555-7763)\n2. Visit www.example.com/products\n3. Speak with a representative at any branch location\n\nOur team can provide personalized product recommendations based on your financial needs.",
            "contact_info": {
                "phone": "1-800-555-7763",
                "website": "www.example.com/products"
            }
        },
        "account_inquiry": {
            "response": "For security reasons and to protect your personal information, I'm unable to process account inquiries at this time. Please use one of these secure alternatives:\n\n1. Log in to your online banking portal\n2. Call our secure account services line at 1-800-555-ACCT (1-800-555-2228)\n3. Visit a branch with valid photo identification\n\nFor urgent account matters, our phone representatives are available 24/7.",
            "contact_info": {
                "phone": "1-800-555-2228",
                "online": "Online banking portal",
                "hours": "24/7"
            }
        },
        "compliance": {
            "response": "I apologize, but I cannot provide regulatory or compliance information at this time. For accurate compliance-related inquiries:\n\n1. Contact our compliance department at compliance@example.com\n2. Call our compliance hotline at 1-800-555-CMPL (1-800-555-2675)\n3. Refer to official disclosures at www.example.com/disclosures\n\nAll financial products are subject to terms and conditions. Please consult official documentation.",
            "contact_info": {
                "email": "compliance@example.com",
                "phone": "1-800-555-2675",
                "website": "www.example.com/disclosures"
            }
        },
        "personalized_outreach": {
            "response": "Thank you for your interest. Unfortunately, I'm unable to provide personalized recommendations at this time. To discuss financial solutions tailored to your needs:\n\n1. Schedule an appointment with a financial advisor\n2. Call 1-800-555-ADVS (1-800-555-2387)\n3. Request a callback at www.example.com/contact\n\nOur advisors can help you achieve your financial goals with personalized guidance.",
            "contact_info": {
                "phone": "1-800-555-2387",
                "website": "www.example.com/contact"
            }
        }
    }

    # Get appropriate response
    default_response = responses["general"]
    response_data = responses.get(use_case, default_response)

    # Add standard disclaimer for financial services
    disclaimer = "\n\n---\nThis is an automated response. Products and services are subject to terms and conditions. Not all products are available in all areas. Member FDIC. Equal Housing Lender."

    # Construct final response
    final_response = response_data["response"] + disclaimer

    return {
        'statusCode': 200,
        'body': json.dumps({
            'response': final_response,
            'model_used': 'DEGRADED_SERVICE',
            'use_case': use_case,
            'degradation_reason': reason,
            'contact_info': response_data["contact_info"],
            'timestamp': datetime.now().isoformat(),
            'message': 'Service degraded - using predefined response'
        })
    }
