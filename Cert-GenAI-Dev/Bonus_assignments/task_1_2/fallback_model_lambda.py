
import boto3
import json
import os

bedrock_runtime = boto3.client('bedrock-runtime')

def lambda_handler(event, context):
    """
    Fallback model handler that uses a simpler, more reliable model.

    This Lambda is invoked when the primary model fails, using
    conservative settings for maximum reliability.
    """
    try:
        # Extract parameters
        prompt = event.get('prompt', '')
        use_case = event.get('use_case', 'general')
        is_fallback = event.get('is_fallback', False)

        # Use Titan Express as fallback (simpler, more reliable)
        model_id = "amazon.titan-text-express-v1"

        # Conservative parameters for reliability
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 300,  # Reduced for faster response
                "temperature": 0.5,    # Lower temperature for consistency
                "topP": 0.9,
                "stopSequences": []
            }
        })

        # Invoke model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body
        )

        # Parse response
        response_body = json.loads(response['body'].read().decode())
        output = response_body['results'][0]['outputText']

        return {
            'statusCode': 200,
            'body': json.dumps({
                'response': output,
                'model_used': f"FALLBACK:{model_id}",
                'use_case': use_case,
                'is_fallback': True,
                'message': 'Fallback model used due to primary model failure'
            })
        }

    except Exception as e:
        # Even fallback failed - let Step Functions handle
        raise Exception(f"Fallback model failed: {str(e)}")
