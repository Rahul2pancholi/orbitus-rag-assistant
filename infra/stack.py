"""S3 (documents + per-tenant index) -> Lambda (FastAPI + bundled embedding model) -> Groq LLM, via a Function URL.

Bedrock is not available on the AWS Free plan, so embeddings run inside the Lambda
and answers come from Groq. The Groq key is an SSM SecureString created outside
CloudFormation (CloudFormation cannot create SecureStrings), see README.
"""

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from constructs import Construct

LAMBDA_BUILD_DIR = Path(__file__).resolve().parent.parent / "build" / "lambda"


class RagAssistantStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        tenant_id = self.node.try_get_context("tenant_id") or "demo"
        groq_key_param = self.node.try_get_context("groq_api_key_param") or "/orbitus/groq-api-key"

        bucket = s3.Bucket(
            self,
            "DocsBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=True,  # every index write is recoverable
            removal_policy=RemovalPolicy.DESTROY,  # demo stack; RETAIN in production
            auto_delete_objects=True,
        )

        fn = lambda_.Function(
            self,
            "ApiFunction",
            runtime=lambda_.Runtime.PYTHON_3_13,
            architecture=lambda_.Architecture.ARM_64,
            handler="app.main.handler",
            code=lambda_.Code.from_asset(str(LAMBDA_BUILD_DIR)),
            memory_size=2048,  # more memory = more CPU for the ONNX embedding model
            timeout=Duration.seconds(60),
            log_group=logs.LogGroup(
                self, "ApiLogs", retention=logs.RetentionDays.TWO_WEEKS, removal_policy=RemovalPolicy.DESTROY
            ),
            environment={
                "EMBEDDINGS_PROVIDER": "local",
                "EMBED_MODEL_PATH": "/var/task/embed-model",
                "HF_HUB_OFFLINE": "1",
                "LLM_PROVIDER": "groq",
                "GROQ_API_KEY_PARAM": groq_key_param,
                "STORE_URI": f"s3://{bucket.bucket_name}",
                "TENANT_ID": tenant_id,
            },
        )

        # Least privilege: the function can only touch its own tenant's prefix
        # (read for queries, write for the upload endpoint).
        bucket.grant_read_write(fn, f"tenants/{tenant_id}/*")

        fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["ssm:GetParameter"],
                resources=[f"arn:aws:ssm:{self.region}:{self.account}:parameter/{groq_key_param.lstrip('/')}"],
            )
        )

        # Public URL for the demo (the brief says no auth is required). Production:
        # AuthType.AWS_IAM behind CloudFront + WAF, or API Gateway with a Cognito authorizer.
        url = fn.add_function_url(auth_type=lambda_.FunctionUrlAuthType.NONE)

        CfnOutput(self, "AppUrl", value=url.url)
        CfnOutput(self, "DocsBucketName", value=bucket.bucket_name)
        CfnOutput(self, "LogGroup", value=fn.log_group.log_group_name)
