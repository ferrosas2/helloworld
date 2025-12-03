
import boto3
import json
import os
from datetime import datetime

bedrock_runtime = boto3.client('bedrock-runtime')
appconfig_client = boto3.client('appconfig')

# Cache configuration
config_cache = {
    "data": None,
    "timestamp": None,
    "ttl": 300  # 5 minutes
}

def get_configuration():
    """Retrieve configuration from AppConfig with caching."""
    current_time = datetime.now().timestamp()

    # Check cache
    if (config_cache["data"] is not None and 
        config_cache["timestamp"] is not None and
        current_time - config_cache["timestamp"] < config_cache["ttl"]):
        return config_cache["data"]

    # Fetch fresh configuration
    try:
        response = appconfig_client.get_configuration(
            Application=os.environ['APP_NAME'],
            Environment=os.environ['ENVIRONMENT'],
            Configuration='ModelSelectionStrategy',
            ClientId='model-abstraction-lambda'
        )

        config = json.loads(response['Content'].read().decode('utf-8'))

        # Update cache
        config_cache["data"] = config
        config_cache["timestamp"] = current_time

        return config
    except Exception as e:
        print(f"Error fetching configuration: {str(e)}")
        # Return cached data if available
        return config_cache["data"]

def select_model(config, use_case):
    """Select appropriate model based on configuration and use case."""
    if not config:
        return None

    # Check for use case specific model
    use_case_models = config.get('use_case_models', {})
    if use_case in use_case_models:
        return use_case_models[use_case]

    # Default to primary model
    return config.get('primary_model')

def invoke_model(model_id, prompt, max_tokens=500):
    """Invoke the selected Bedrock model."""
    try:
        # Prepare request based on model provider
        if "anthropic" in model_id:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "top_p": 0.9
            })
        elif "amazon" in model_id:
            body = json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": 0.7,
                    "topP": 0.9
                }
            })
        else:
            raise ValueError(f"Unsupported model: {model_id}")

        # Invoke model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body
        )

        # Parse response
        response_body = json.loads(response['body'].read().decode())

        if "anthropic" in model_id:
            output = response_body['content'][0]['text']
        elif "amazon" in model_id:
            output = response_body['results'][0]['outputText']

        return {
            "success": True,
            "output": output,
            "model_used": model_id
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model_used": model_id
        }

def lambda_handler(event, context):
    """Main Lambda handler for model abstraction."""
    try:
        # Parse request
        body = json.loads(event.get('body', '{}'))
        prompt = body.get('prompt', '')
        use_case = body.get('use_case', 'general')
        max_tokens = body.get('max_tokens', 500)

        # Validate input
        if not prompt:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Prompt is required'})
            }

        # Get configuration
        config = get_configuration()
        if not config:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Failed to load configuration'})
            }

        # Select model
        model_id = select_model(config, use_case)
        if not model_id:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'No model available for use case'})
            }

        # Invoke model with retries
        max_retries = config.get('guardrails', {}).get('max_retries', 3)

        for attempt in range(max_retries):
            result = invoke_model(model_id, prompt, max_tokens)

            if result["success"]:
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'response': result["output"],
                        'model_used': result["model_used"],
                        'use_case': use_case,
                        'attempt': attempt + 1
                    }),
                    'headers': {
                        'Content-Type': 'application/json'
                    }
                }

        # All retries failed
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Model invocation failed after retries',
                'model_used': model_id
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
