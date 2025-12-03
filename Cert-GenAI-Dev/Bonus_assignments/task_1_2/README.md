# Resilient Financial Services AI Assistant

This repository contains the implementation for the AWS Certified Generative AI Developer bonus assignment (Task 1.2). It provides a comprehensive blueprint for building a resilient, compliant, and cost-effective AI assistant for financial services using Amazon Bedrock.

The solution addresses the unique challenges of using Large Language Models (LLMs) in regulated industries by implementing benchmarking, dynamic routing, circuit breakers, and lifecycle management.

## Notebook Structure

The implementation is contained in `financials_ai_assistant.ipynb` and is organized into the following sections:

### Part 1: Foundation Model Benchmarking

This section establishes a baseline for performance, cost, and quality by evaluating different Amazon Bedrock models against financial domain tasks.

- **Section 1: Initialize AWS Clients and Configuration**
  - Sets up Boto3 clients for Bedrock Runtime, AppConfig, Lambda, and CloudWatch.
  - Defines global configuration variables including `PRIMARY_REGION` (us-east-1) and the list of `MODELS_TO_EVALUATE` (e.g., Claude 3 Sonnet, Claude Instant, Titan Text Express).

- **Section 2: Setup Evaluation Framework**
  - Implements the `invoke_model` function to standardize interactions with different model providers (Anthropic vs. Amazon).
  - Handles request formatting and response parsing, ensuring a consistent interface for evaluation.
  - Captures raw performance metrics such as latency and token counts for every invocation.

- **Section 3: Define Test Cases and Metrics**
  - Establishes a suite of `FINANCIAL_TEST_CASES` covering various use cases like `product_question`, `compliance`, and `account_inquiry`.
  - Implements `calculate_similarity` to score model outputs against ground truth answers using Jaccard similarity and recall metrics.

- **Section 4: Run Model Evaluation**
  - Executes the benchmark by iterating through every combination of model and test case.
  - Collects comprehensive metrics (latency, similarity score, estimated cost) into a Pandas DataFrame.
  - Generates a summary report comparing models across key performance indicators.

- **Section 5: Analyze Results and Create Selection Strategy**
  - Processes evaluation data to generate a `model_selection_strategy.json`.
  - Normalizes scores for quality, latency, and cost to calculate an overall weighted score.
  - Assigns the optimal model to each specific use case (e.g., high-intelligence models for compliance, faster models for general inquiries).

### Part 2: Flexible Architecture for Dynamic Model Selection

This section details the implementation of a dynamic routing layer that decouples client applications from specific model providers.

- **Section 6: Configure AWS AppConfig**
  - Initializes the AppConfig infrastructure (`AIAssistantApp`, `prod` environment, and `ModelSelectionStrategy` profile).
  - Deploys the generated strategy JSON as a hosted configuration version.
  - Sets up a deployment strategy (e.g., `FastDeployment`) to ensure immediate propagation of configuration changes.

- **Section 7: Implement Model Abstraction Lambda**
  - Generates the `model_abstraction_lambda.py` code which acts as the central gateway.
  - Implements a caching mechanism (`config_cache`) with a TTL to minimize AppConfig API calls.
  - Contains logic to fetch the configuration, select the appropriate model for the `use_case`, and handle provider-specific request/response formats.

- **Section 8: Deploy API Gateway Integration**
  - Creates a regional REST API (`AIAssistantApp-API`) with a `/generate` resource.
  - Configures a `POST` method with **Lambda Proxy Integration**, allowing the Lambda to fully control the HTTP response.
  - Deploys the API to the `prod` stage and outputs the invocation URL.

### Part 3: Resilient System Design

This section implements fault-tolerant patterns to ensure service continuity even during failures.

- **Section 9: Create Step Functions Circuit Breaker**
  - Defines a state machine (`circuit_breaker_state_machine.json`) implementing the circuit breaker pattern.
  - Tracks failure counts in DynamoDB and opens the circuit after 5 consecutive failures.
  - Automatically retries with exponential backoff before falling back to secondary models.
  - Includes states: TryPrimaryModel, IncrementFailureCount, CheckCircuitBreaker, TryFallbackModel, GracefulDegradation, ResetFailureCount.

- **Section 10: Implement Fallback Lambda**
  - Creates `fallback_model_lambda.py` using Amazon Titan Express for maximum reliability.
  - Uses conservative parameters (reduced max tokens: 300, lower temperature: 0.5) to prioritize consistency.
  - Marks responses with `FALLBACK:` prefix for monitoring and debugging.
  - Propagates errors to Step Functions for graceful degradation if fallback fails.

- **Section 11: Implement Graceful Degradation Lambda**
  - Creates `graceful_degradation_lambda.py` for static, regulation-compliant responses.
  - Provides predefined responses by use case (general, product_question, account_inquiry, compliance, personalized_outreach).
  - Includes contact information (phone numbers, websites, email addresses) for each scenario.
  - Adds standard financial disclaimers (FDIC, Equal Housing Lender) to all responses.
  - Marks responses with `DEGRADED_SERVICE` for monitoring.

- **Section 12: Setup Cross-Region Deployment with CloudFormation**
  - Generates `cross_region_deployment.yaml` CloudFormation template for multi-region high availability.
  - Creates Lambda functions (ModelAbstraction, FallbackModel, GracefulDegradation) with proper IAM roles.
  - Sets up DynamoDB table for circuit breaker state with point-in-time recovery.
  - Configures API Gateway with Lambda Proxy Integration.
  - Includes CloudWatch Log Groups for centralized logging.
  - Supports deployment to both `us-east-1` (primary) and `us-west-2` (secondary).

- **Section 13: Configure Route 53 Failover Routing**
  - Implements DNS-based failover using Route 53 health checks.
  - Creates health check for primary region (HTTPS on port 443, 30s interval, 3 failure threshold).
  - Automatically redirects traffic to secondary region if primary region fails.
  - Configures PRIMARY and SECONDARY failover record sets.
  - Supports custom domain configuration with hosted zone management.

### Part 4: Model Customization & Lifecycle Management

This section covers fine-tuning custom models and managing their lifecycle in production.

- **Step 1: Fine-tune a model with SageMaker**
  - Creates a financial Q&A dataset (`financial_qa_dataset.csv`) with 5 domain-specific examples.
  - Generates `train.py` training script using HuggingFace Transformers and Trainer API.
  - Uploads dataset to S3 (`s3://cert-genai-dev/bonus_1_2/data/`).
  - Configures SageMaker HuggingFace Estimator with CPU-compatible versions:
    - `transformers_version="4.26.0"`
    - `pytorch_version="1.13.1"`
    - `py_version="py39"`
    - `instance_type="ml.m5.xlarge"` (CPU instance to avoid GPU quota limits)
  - Automatically searches for SageMaker execution role in IAM.
  - Note: Actual training job execution requires GPU quota increase or SDK upgrade.

- **Step 2: Deploy the Fine-tuned Model**
  - Implements `deploy_fine_tuned_model()` function to create SageMaker endpoints.
  - Retrieves model artifacts from completed training jobs via `describe_training_job()`.
  - Creates SageMaker Model with container configuration and environment variables.
  - Creates endpoint configuration with `ml.m5.xlarge` instances and production variants.
  - Deploys endpoint and waits for `InService` status (5-10 minutes).
  - Returns endpoint details including name, configuration, model name, and status.

- **Step 3: Test the Deployed Model**
  - Implements `test_sagemaker_endpoint()` function for automated testing.
  - Invokes endpoint with financial domain questions using `invoke_endpoint()`.
  - Supports custom test questions or uses default financial Q&A set.
  - Collects success rates, answers, and timestamps for each test.
  - Provides detailed summary with successful vs failed invocations.

- **Step 4: Implement Model Monitoring with CloudWatch**
  - Implements `setup_model_monitoring()` function to configure CloudWatch monitoring.
  - Creates three CloudWatch alarms:
    - **High Latency Alarm**: Triggers when average latency exceeds 5 seconds (2 evaluation periods).
    - **High Error Rate Alarm**: Triggers when 4XX errors exceed 10 in 5 minutes.
    - **High CPU Utilization Alarm**: Triggers when CPU exceeds 80% (2 evaluation periods).
  - Creates CloudWatch dashboard with three widgets:
    - Model Latency (Average and p99 metrics)
    - Invocations and Errors (Invocations, 4XX errors, 5XX errors)
    - Resource Utilization (CPU and Memory utilization)
  - Provides console URL for real-time monitoring and visualization.

- **Step 5: Model Lifecycle Management**
  - **A/B Testing**: `implement_ab_testing()` function implements traffic splitting between model variants.
    - Updates endpoint configuration with two production variants (VariantA and VariantB).
    - Configures traffic distribution (e.g., 90% current model, 10% new model).
    - Creates variant-specific CloudWatch metrics for performance comparison.
    - Allows gradual rollout of new models based on monitored performance.
    - Waits for endpoint update to complete before activating A/B test.
  - **Resource Cleanup**: `cleanup_old_resources()` function manages AWS costs.
    - Lists endpoints, endpoint configs, and models older than specified days (default: 7).
    - Supports dry-run mode to preview deletions before execution.
    - Automatically removes unused resources to reduce AWS costs.
    - Provides cleanup summary with counts for each resource type.
    - Respects cutoff date calculation using timedelta.

## Deployment & Testing

### Prerequisites
- AWS Credentials configured with appropriate permissions.
- Python 3.11+ environment.
- Required packages: `boto3`, `pandas`, `numpy`, `sagemaker`.
- SageMaker execution role with permissions for S3, Bedrock, CloudWatch, and SageMaker.

### Installation
```bash
pip install boto3 pandas numpy sagemaker
```

### Running the Notebook
1. Open `financials_ai_assistant.ipynb` in Jupyter or VS Code.
2. Execute cells sequentially from Part 1 through Part 4.
3. Review generated files:
   - `model_evaluation_results.csv` - Benchmark results for all models and test cases
   - `model_selection_strategy.json` - Dynamic routing configuration for AppConfig
   - `financial_qa_dataset.csv` - Training dataset for SageMaker fine-tuning
   - `train.py` - HuggingFace training script
   - `model_abstraction_lambda.py` - Primary Lambda function for model routing
   - `fallback_model_lambda.py` - Fallback Lambda using Titan Express
   - `graceful_degradation_lambda.py` - Final safety net with static responses
   - `circuit_breaker_state_machine.json` - Step Functions state machine definition
   - `cross_region_deployment.yaml` - CloudFormation template for multi-region HA

### Testing the API Endpoint
Once the API Gateway is deployed (Section 8), you can test the endpoint using the following PowerShell command:

```powershell
Invoke-RestMethod -Uri "https://qqskzpxhgb.execute-api.us-east-1.amazonaws.com/prod/generate" -Method Post -Body '{"prompt": "What is a 401k?", "use_case": "product_question"}' -ContentType "application/json"
```

*Note: Replace the URI with your specific API Gateway endpoint URL if different.*

### Expected Output
The API should return a JSON response containing the model's answer:

```json
{
    "response": "A 401(k) is a tax-advantaged retirement savings plan...",
    "model_used": "anthropic.claude-3-sonnet-20240229-v1:0",
    "use_case": "product_question",
    "attempt": 1
}
```

### SageMaker Fine-Tuning Configuration

The notebook uses CPU-compatible versions to avoid GPU instance quota limits:

**Current Configuration (CPU-Compatible):**
```python
instance_type = "ml.m5.xlarge"          # CPU instance
transformers_version = "4.26.0"         # Supports CPU
pytorch_version = "1.13.1"              # Supports CPU
py_version = "py39"                     # Python 3.9
```

**Alternative Options:**

1. **Use GPU Instance** (requires quota increase via AWS Service Quotas):
   ```python
   instance_type = "ml.g4dn.xlarge"     # GPU instance
   transformers_version = "4.17.0"
   pytorch_version = "1.10.2"
   py_version = "py38"
   ```

2. **Upgrade SageMaker SDK**:
   ```bash
   pip install --upgrade sagemaker
   ```
   This may resolve CPU compatibility issues with newer SDK versions.

3. **Use Generic Estimator** (not HuggingFace-specific):
   ```python
   from sagemaker.estimator import Estimator
   image_uri = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04"
   ```

## Key Components

### Architecture Layers

1. **Model Abstraction Layer**
   - Decouples client applications from specific model IDs.
   - Enables hot-swapping of models without code changes.
   - Implements retry logic and error handling with configurable max retries.
   - Caches AppConfig configuration with 5-minute TTL for performance.

2. **Dynamic Configuration (AppConfig)**
   - Allows changing the "Primary Model" without code deployments.
   - Supports use case-specific model assignments (e.g., compliance → Claude 3 Sonnet).
   - Provides configuration versioning and rollback capabilities.
   - Implements fast deployment strategy for immediate propagation.

3. **Resilience Patterns**
   - **Circuit Breaker**: Prevents cascading failures during outages with DynamoDB state tracking.
   - **Fallback Models**: Automatic degradation to simpler, more reliable models (Titan Express).
   - **Graceful Degradation**: Static, regulation-compliant responses when all models fail.
   - **Exponential Backoff**: Retry logic with increasing delays (2x backoff rate, max 2 attempts).

4. **High Availability**
   - Cross-region deployment with CloudFormation templates for `us-east-1` and `us-west-2`.
   - Route 53 DNS failover with health checks (30s interval, 3 failure threshold).
   - DynamoDB for distributed circuit breaker state with point-in-time recovery.
   - API Gateway regional endpoints with automatic failover.

5. **Model Lifecycle Management**
   - SageMaker fine-tuning for domain-specific models with HuggingFace Transformers.
   - A/B testing for gradual rollout with traffic splitting (e.g., 90/10 split).
   - CloudWatch monitoring with automated alarms (latency, errors, CPU utilization).
   - Automated cleanup of old resources (endpoints, configs, models older than 7 days).

### Generated Artifacts

| File | Description | Purpose |
|------|-------------|---------|
| `model_evaluation_results.csv` | Benchmark results for all models and test cases | Model selection and performance analysis |
| `model_selection_strategy.json` | Dynamic routing configuration for AppConfig | Runtime model selection without redeployment |
| `financial_qa_dataset.csv` | Training dataset with 5 Q&A pairs | SageMaker fine-tuning input |
| `train.py` | HuggingFace training script with Trainer API | Custom model training logic |
| `model_abstraction_lambda.py` | Primary Lambda function for model routing | Central gateway with caching and retries |
| `fallback_model_lambda.py` | Fallback Lambda using Titan Express | Secondary model for reliability |
| `graceful_degradation_lambda.py` | Final safety net with static responses | Regulation-compliant fallback responses |
| `circuit_breaker_state_machine.json` | Step Functions state machine definition | Automated failure handling and retries |
| `cross_region_deployment.yaml` | CloudFormation template for multi-region HA | Infrastructure as code for deployment |

## Features Implemented

### ✅ Foundation Model Benchmarking
- Evaluated Claude 3 Sonnet, Claude Instant, and Titan Text Express across 7 financial test cases.
- Measured **latency** (ms), **quality** (Jaccard + recall similarity scores), and **estimated cost**.
- Created data-driven model selection strategy with weighted overall scores (50% quality, 30% latency, 20% cost).
- Assigned optimal models to specific use cases based on performance metrics.

### ✅ Flexible Architecture
- AWS AppConfig for dynamic model routing without redeployment.
- Lambda-based model abstraction layer with 5-minute configuration caching.
- API Gateway REST API integration with `/generate` endpoint and Lambda Proxy mode.
- Supports multiple model providers (Anthropic Claude, Amazon Titan) with unified interface.

### ✅ Resilient System Design
- Step Functions circuit breaker pattern with DynamoDB state tracking (opens after 5 failures).
- Automatic fallback to secondary models (Titan Express with conservative parameters).
- Graceful degradation with regulation-compliant static responses by use case.
- CloudFormation templates for automated cross-region deployment.
- Route 53 DNS failover with health checks (HTTPS, SNI enabled, 30s interval).

### ✅ Model Customization & Lifecycle
- SageMaker fine-tuning with HuggingFace Transformers on financial Q&A dataset.
- CPU-compatible configuration (`ml.m5.xlarge`, PyTorch 1.13.1, Transformers 4.26.0).
- Model deployment to real-time inference endpoints with automatic status monitoring.
- CloudWatch monitoring with 3 automated alarms (latency >5s, errors >10/5min, CPU >80%).
- A/B testing for gradual model rollout with configurable traffic splitting.
- Automated resource cleanup to manage AWS costs (dry-run and execution modes).

## Cost Optimization

The solution includes several cost optimization strategies:

1. **Dynamic Model Selection**: Routes simple queries to cheaper models (Titan Express ~$0.001/1K tokens) and complex queries to premium models (Claude 3 Sonnet ~$0.002/1K tokens).
2. **Resource Cleanup**: Automatically identifies and deletes endpoints, configs, and models older than 7 days (configurable).
3. **CPU Instances**: Uses `ml.m5.xlarge` ($0.269/hour) for SageMaker training instead of GPU instances ($0.736/hour for ml.g4dn.xlarge).
4. **AppConfig Caching**: Implements 5-minute TTL to minimize AppConfig API calls ($0.08 per 10,000 requests).
5. **Circuit Breaker**: Prevents wasted API calls during outages by opening circuit after 5 consecutive failures.
6. **Pay-per-Request DynamoDB**: Uses on-demand billing mode for circuit breaker state table.

## Compliance & Governance

The solution addresses financial services compliance requirements:

- **Regulation-Compliant Responses**: Graceful degradation includes FDIC disclaimers, Equal Housing Lender notices, and "Not financial advice" disclaimers.
- **Audit Trail**: All model invocations logged with timestamps, model IDs, use cases, and attempt numbers.
- **Contact Information**: Provides phone numbers (1-800-555-xxxx), website links, and email addresses when AI fails.
- **Data Privacy**: No customer data stored; fully stateless architecture with no PII retention.
- **Model Versioning**: CloudWatch tracks which model version (Primary, Fallback, Degraded) served each request.
- **Failover Documentation**: Route 53 health checks and Step Functions provide transparent failure handling.

## Next Steps

To deploy this solution to production:

1. **Request GPU Quota**: Submit a request via AWS Service Quotas for `ml.g4dn.xlarge` instances if GPU training is needed.
2. **Custom Domain**: Configure Route 53 with your registered domain (e.g., `ai-assistant.yourdomain.com`) for the API Gateway.
3. **SNS Notifications**: Create SNS topics and update CloudWatch alarms to send email/SMS notifications (set `ActionsEnabled=True`).
4. **Authentication**: Add API Gateway API keys (`x-api-key` header) or AWS Cognito for user authentication and rate limiting.
5. **Model Drift Detection**: Implement SageMaker Model Monitor to detect data drift and quality degradation in production.
6. **CI/CD Pipeline**: Use AWS CodePipeline, CodeBuild, or GitHub Actions for automated testing and deployment.
7. **Load Testing**: Use AWS Distributed Load Testing or Apache JMeter to validate scalability under peak loads.
8. **Logging & Tracing**: Integrate AWS X-Ray for distributed tracing and CloudWatch Logs Insights for log analysis.

## Troubleshooting

### Common Issues

1. **SageMaker Role Not Found**
   - **Problem**: `ValueError: Could not get execution role from environment`
   - **Solution**: Create a SageMaker execution role in IAM console with `AmazonSageMakerFullAccess` policy. Ensure role has trust relationship with `sagemaker.amazonaws.com`.

2. **GPU Instance Quota Exceeded**
   - **Problem**: `ResourceLimitExceeded: The account-level service limit 'ml.g4dn.xlarge for endpoint usage' is 0 Instances`
   - **Solution**: Use CPU configuration (`ml.m5.xlarge`, `transformers_version="4.26.0"`, `pytorch_version="1.13.1"`) OR request quota increase via AWS Service Quotas console.

3. **Unsupported Processor Error**
   - **Problem**: `ValueError: Unsupported processor: cpu. This model is only supported on gpu. Supported processor(s): gpu`
   - **Solution**: Update to CPU-compatible versions (Transformers 4.26.0, PyTorch 1.13.1, py39) OR use GPU instance with quota.

4. **AppConfig Configuration Not Found**
   - **Problem**: `ClientException: Configuration not found`
   - **Solution**: Ensure `model_selection_strategy.json` exists before running Section 6. Verify AppConfig application, environment, and profile were created successfully.

5. **API Gateway 403 Forbidden**
   - **Problem**: `{"message": "Missing Authentication Token"}` or 403 errors
   - **Solution**: Verify Lambda permission allows `apigateway.amazonaws.com` to invoke function. Check API Gateway deployment stage matches `ENVIRONMENT` variable (`prod`).

6. **Circuit Breaker DynamoDB Errors**
   - **Problem**: `ResourceNotFoundException: Requested resource not found`
   - **Solution**: Ensure DynamoDB table (`AIAssistantApp-CircuitBreaker-prod`) exists. Verify Lambda execution role has `dynamodb:UpdateItem` and `dynamodb:GetItem` permissions.

## License

This project is provided as-is for educational purposes as part of the AWS Certified Generative AI Developer certification.

## Author

Created for AWS Certified Generative AI Developer - Bonus Assignment (Task 1.2)

---

For questions or issues, please refer to the inline documentation in `financials_ai_assistant.ipynb`.
